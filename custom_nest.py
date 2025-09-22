import tensorflow as tf
import scipy
import numpy as np


class GatherEmbeddings(tf.Module):
    def __init__(self, nmod, nvec, R):
        self.nmod = nmod
        self.nvec = nvec
        self.R = R
        super().__init__()
        self.generate_stick_breaking_weights()

    def generate_stick_breaking_weights(self):
        self.tf_v_hat = [None] * self.nmod
        for k in range(self.nmod):
            self.tf_v_hat[k] = tf.Variable(
                scipy.special.logit(
                    np.random.rand(self.nvec[k], self.R).astype(np.float32)
                ),
                name=f"embeddings_tf_v_hat_{k}",
            )

    def get_trainable_variables(self):
        return self.tf_v_hat

    def generate_embeddings(self):
        self.log_omega = [None] * self.nmod
        self.tf_U = [None] * self.nmod
        for k in range(self.nmod):
            log_v = tf.math.log_sigmoid(self.tf_v_hat[k])
            log_v_minus = tf.math.log_sigmoid(-self.tf_v_hat[k])
            cum_sum = tf.cumsum(log_v_minus, exclusive=True, axis=1)
            self.log_omega[k] = log_v + cum_sum
            self.tf_U[k] = tf.exp(self.log_omega[k])

    def __call__(self, indices):
        self.generate_embeddings()
        return tf.concat(
            [
                tf.gather(self.tf_U[0], indices[:, 0]),
                tf.gather(self.tf_U[1], indices[:, 1]),
                tf.gather(self.tf_U[2], indices[:, 2]),
                tf.gather(self.tf_U[3], indices[:, 3]),
            ],
            1,
        )


class RFFTransform(tf.Module):
    def __init__(self, nmod, R, m):
        self.nmod = nmod
        self.R = R
        self.m = m
        super().__init__()
        self.setup_params()

    def setup_params(self):
        self.d = self.nmod * self.R
        self.tf_S = tf.Variable(
            (np.random.randn(self.m, self.d) / (2 * np.pi)).astype(np.float32),
            name="rff_transform_tf_S",
        )
        self.b = tf.Variable(
            np.random.uniform(0, 2 * np.pi, size=(1, self.m)).astype(np.float32),
            name="rff_transform_b",
        )

    def get_trainable_variables(self):
        return [self.tf_S, self.b]

    def __call__(self, embeddings):
        sub_phi_lin = tf.matmul(embeddings, tf.transpose(self.tf_S)) + self.b
        sub_phi = tf.concat([tf.cos(sub_phi_lin), tf.sin(sub_phi_lin)], 1)
        return sub_phi


class GaussianProcess(tf.Module):
    def __init__(self, m):
        self.m = m
        super().__init__()
        self.setup_params()

    def setup_params(self):
        self.w_mu = tf.Variable(
            np.random.rand(2 * self.m, 1).astype(np.float32),
            name="gaussian_process_w_mu",
        )
        # self.w_L = tf.Variable(1.0 / self.m * np.eye(2 * self.m))
        # self.w_Ltril = tf.matrix_band_part(self.w_L, -1, 0)

    def get_trainable_variables(self):
        return [self.w_mu]
        # return [self.w_mu, self.w_L]

    def __call__(self, features):
        self.score_logits_node = tf.matmul(features, self.w_mu)
        return tf.sigmoid(self.score_logits_node)


class NEST2(tf.keras.Model):
    def __init__(self, nmod, nvec, R, m):
        self.nmod = nmod
        self.nvec = nvec
        self.R = R
        self.m = m
        super().__init__()
        self.setup_params()

    def setup_params(self):
        self.gather_embeddings = GatherEmbeddings(self.nmod, self.nvec, self.R)
        self.rff_transform = RFFTransform(self.nmod, self.R, self.m)
        self.gaussian_process = GaussianProcess(self.m)

    @property
    def trainable_variables(self):
        return (
            self.gather_embeddings.get_trainable_variables()
            + self.rff_transform.get_trainable_variables()
            + self.gaussian_process.get_trainable_variables()
        )

    def call(self, indices):
        embeddings = self.gather_embeddings(indices)
        features = self.rff_transform(embeddings)
        probs = self.gaussian_process(features)
        return probs

    def compute_loss(self, indices, labels):
        probs = self(indices)
        loss = tf.keras.ops.binary_crossentropy(labels, probs)
        return tf.reduce_mean(loss)

    def train_step(self, xy):
        indices, labels = xy
        with tf.GradientTape() as tape:
            loss = self.compute_loss(indices, labels)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for m in self.metrics:
            if m.name == "loss":
                m.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def compute_mrr(self, x, n_entities):
        samples = 50
        xs = x[:samples, :-1]
        ys = x[:samples, -1]
        """
        mrr = np.zeros(samples)
        for i in range(samples):
            p = np.tile(xs[i], (n_entities, 1))
            p = np.hstack([p, np.arange(n_entities).reshape((-1, 1))])
            probs = self(p)
            probs = tf.reshape(probs, (n_entities,))
            args = tf.argsort(probs)
            rank = tf.where(args == ys[i])[0, 0] + 1
            mrr[i] = rank.numpy()
        mrr = np.mean(1.0 / mrr)
        """
        xs = np.tile(xs, (1, n_entities)).reshape((-1, 3))
        es = np.tile(np.arange(n_entities), (samples)).reshape((-1, 1))
        xs = np.hstack([xs, es])
        probs = self.predict(xs)
        probs = probs.reshape((samples, n_entities))
        args = np.argsort(probs, axis=1)
        ranks = np.where(args == ys[:, None])[1] + 1
        mrr = np.mean(1.0 / ranks)
        return mrr

    def compute_hit10(self, x, n_entities):
        samples = 50
        xs = x[:samples, :-1]
        ys = x[:samples, -1]
        xs = np.tile(xs, (1, n_entities)).reshape((-1, 3))
        es = np.tile(np.arange(n_entities), (samples)).reshape((-1, 1))
        xs = np.hstack([xs, es])
        probs = self.predict(xs)
        probs = probs.reshape((samples, n_entities))
        args = np.argsort(probs, axis=1)
        ranks = np.where(args == ys[:, None])[1] + 1
        hit10 = np.mean(ranks <= 10)
        return hit10

    def sample_prediction(self, x, n_entities):
        xs = x[:1, :-1]
        ys = x[:1, -1]
        xs = np.tile(xs, (1, n_entities)).reshape((-1, 3))
        es = np.tile(np.arange(n_entities), (1)).reshape((-1, 1))
        xs = np.hstack([xs, es])
        probs = self.predict(xs)
        probs = probs.reshape((1, n_entities))[0]
        print(ys, probs[ys[0]])
        print(probs.min(), probs.max())
        print(probs.argmin(), probs.argmax())
        print(probs.mean(), probs.std())


class LoadData:
    def __init__(self, path, negative_ratio=1):
        self.path = path
        self.negative_ratio = negative_ratio
        self.entity_mapping = {}
        self.relation_mapping = {}
        self.train_data_x, self.train_data_y, self.nvec = self.load_data(
            self.path + "/train.txt"
        )
        self.test_data_x, self.test_data_y, _ = self.load_data(self.path + "/test.txt")

    def generate_negative_samples(self, data):
        N = len(data)
        for i in range(N):
            for _ in range(self.negative_ratio):
                mode = np.random.randint(1, 4)
                sample = data[i].copy()
                sample[mode] = np.random.randint(0, len(self.entity_mapping))
                sample[-1] = 0
                data.append(sample)

    def load_data(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                parts = line.split(" ")
                if len(parts) < 4:
                    continue
                r, e1, e2, e3 = parts[:4]
                if r not in self.relation_mapping:
                    self.relation_mapping[r] = len(self.relation_mapping)
                if e1 not in self.entity_mapping:
                    self.entity_mapping[e1] = len(self.entity_mapping)
                if e2 not in self.entity_mapping:
                    self.entity_mapping[e2] = len(self.entity_mapping)
                if e3 not in self.entity_mapping:
                    self.entity_mapping[e3] = len(self.entity_mapping)
                data.append(
                    [
                        self.relation_mapping[r],
                        self.entity_mapping[e1],
                        self.entity_mapping[e2],
                        self.entity_mapping[e3],
                        1,
                    ]
                )
        self.generate_negative_samples(data)
        d = np.array(data, dtype=np.int32)
        np.random.shuffle(d)
        return (
            d[:, :-1],
            d[:, -1:].astype(np.float32),
            [
                len(self.relation_mapping),
                len(self.entity_mapping),
                len(self.entity_mapping),
                len(self.entity_mapping),
            ],
        )


data = LoadData("data/WikiPeople-3", negative_ratio=2)
model = NEST2(nmod=4, nvec=data.nvec, R=10, m=100)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

try:
    model = tf.keras.models.load_model("model.h5")
except:
    pass

while True:
    model.fit(
        data.train_data_x,
        data.train_data_y,
        epochs=10,
        batch_size=16,
    )
    # print(model.compute_mrr(data.test_data_x, data.nvec[1]))
    # print(model.compute_hit10(data.test_data_x, data.nvec[1]))
    model.sample_prediction(data.test_data_x, data.nvec[1])
    model.save("model.h5")
