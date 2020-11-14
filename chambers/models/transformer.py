import tensorflow as tf
import tensorflow_addons as tfa
from einops.layers.tensorflow import Rearrange

from chambers.layers.embedding import PositionalEmbedding1D, ConcatEmbedding, LearnedEmbedding1D
from chambers.layers.transformer import Encoder, Decoder


def Seq2SeqTransformer(input_vocab_size, output_vocab_size, embed_dim, num_heads, dim_feedforward,
                       num_encoder_layers, num_decoder_layers, dropout_rate=0.1, name="seq2seq_transformer"):
    inputs = tf.keras.layers.Input(shape=(None,), name="inputs_tokens")
    targets = tf.keras.layers.Input(shape=(None,), name="targets_tokens")

    x_enc = tf.keras.layers.Embedding(input_vocab_size, embed_dim, mask_zero=True, name="inputs_embed")(inputs)
    x_enc = PositionalEmbedding1D(embed_dim, name="inputs_positional_encoding")(x_enc)
    x = Encoder(embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=dim_feedforward,
                num_layers=num_encoder_layers,
                dropout_rate=dropout_rate)(x_enc)

    x_dec = tf.keras.layers.Embedding(output_vocab_size, embed_dim, mask_zero=True, name="targets_embed")(targets)
    x_dec = PositionalEmbedding1D(embed_dim, name="targets_positional_encoding")(x_dec)
    x = Decoder(embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=dim_feedforward,
                num_layers=num_decoder_layers,
                dropout_rate=dropout_rate,
                norm=False,
                causal=True)([x_dec, x])

    x = tf.keras.layers.Dense(output_vocab_size)(x)

    model = tf.keras.models.Model(inputs=[inputs, targets], outputs=x, name=name)
    return model


def VisionTransformer(input_shape, n_classes, patch_size, patch_dim, n_encoder_layers, n_heads, ff_dim,
                      dropout_rate=0.0):
    inputs = tf.keras.layers.Input(input_shape)
    x = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)(inputs)
    x = tf.keras.layers.Dense(patch_dim)(x)
    x = ConcatEmbedding(1, patch_dim,
                        side="left",
                        axis=1,
                        initializer=tf.keras.initializers.RandomNormal(),
                        name="add_cls_token")(x)
    x = LearnedEmbedding1D(x.shape[1], patch_dim,
                           initializer=tf.keras.initializers.RandomNormal(),
                           name="pos_embedding")(x)
    x = Encoder(embed_dim=patch_dim,
                num_heads=n_heads,
                ff_dim=ff_dim,
                num_layers=n_encoder_layers,
                dropout_rate=dropout_rate)(x)
    x = tf.keras.layers.Cropping1D((0, x.shape[1] - 1))(x)
    x = tf.keras.layers.Reshape([-1])(x)

    x = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation=tfa.activations.gelu),
        tf.keras.layers.Dense(n_classes)],
        name="mlp_head")(x)

    model = tf.keras.models.Model(inputs, x)
    return model


def VisionTransformerOS(input_shape, patch_size, patch_dim, n_encoder_layers, n_heads, ff_dim, dropout_rate=0.0):
    inputs1 = tf.keras.layers.Input(input_shape, name="x1")
    inputs2 = tf.keras.layers.Input(input_shape, name="x2")

    x1 = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)(inputs1)
    x2 = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)(inputs2)

    x1 = tf.keras.layers.Dense(patch_dim)(x1)
    x2 = tf.keras.layers.Dense(patch_dim)(x2)

    x = ConcatEmbedding(n_embeddings=1,
                        embedding_dim=patch_dim,
                        side="left",
                        axis=1,
                        initializer=tf.keras.initializers.RandomNormal(),
                        name="add_cls_token")(x1)
    x = ConcatEmbedding(n_embeddings=1,
                        embedding_dim=patch_dim,
                        side="right",
                        axis=1,
                        initializer=tf.keras.initializers.RandomNormal(),
                        name="add_sep_token")(x)
    x = tf.keras.layers.Concatenate(axis=1)([x, x2])

    x = LearnedEmbedding1D(x.shape[1], patch_dim,
                           initializer=tf.keras.initializers.RandomNormal(),
                           name="pos_embedding")(x)
    x = Encoder(embed_dim=patch_dim,
                num_heads=n_heads,
                ff_dim=ff_dim,
                num_layers=n_encoder_layers,
                dropout_rate=dropout_rate)(x)
    x = tf.keras.layers.Cropping1D((0, x.shape[1] - 1))(x)
    x = tf.keras.layers.Reshape([-1])(x)

    # MLP
    x = tf.keras.layers.Dense(ff_dim, activation=tfa.activations.gelu)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model([inputs1, inputs2], x)
    return model


def VisionTransformerSeg(input_shape, n_classes, patch_size, patch_dim, n_encoder_layers, n_heads,
                         ff_dim, dropout_rate=0.0):
    inputs = tf.keras.layers.Input(input_shape)
    x = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)(inputs)
    x = tf.keras.layers.Dense(patch_dim)(x)
    x = ConcatEmbedding(1, patch_dim,
                        side="left",
                        axis=1,
                        initializer=tf.keras.initializers.RandomNormal(),
                        name="add_cls_token")(x)
    x = LearnedEmbedding1D(x.shape[1], patch_dim,
                           initializer=tf.keras.initializers.RandomNormal(),
                           name="pos_embedding")(x)
    x = Encoder(embed_dim=patch_dim,
                num_heads=n_heads,
                ff_dim=ff_dim,
                num_layers=n_encoder_layers,
                dropout_rate=dropout_rate)(x)

    xc = tf.keras.Sequential([
        tf.keras.layers.Cropping1D((0, x.shape[1] - 1)),
        tf.keras.layers.Reshape([-1]),
        tf.keras.layers.Dense(ff_dim, activation=tfa.activations.gelu),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ], name="class")(x)

    h = int(input_shape[0] / patch_size)
    w = int(input_shape[1] / patch_size)
    xm = tf.keras.Sequential([
        tf.keras.layers.Cropping1D((1, 0)),
        tf.keras.layers.Dense(ff_dim, activation=tfa.activations.gelu),
        tf.keras.layers.Dense(patch_size * patch_size * input_shape[-1], activation=tfa.activations.gelu),
        Rearrange('b (h w) (p1 p2 c) -> b (h p1) (w p2) c', p1=patch_size, p2=patch_size, h=h, w=w),
    ], name="mask")(x)

    model = tf.keras.models.Model(inputs, [xc, xm])
    return model


tf.keras.utils.get_custom_objects().update({
    "Rearrange": Rearrange,
})
