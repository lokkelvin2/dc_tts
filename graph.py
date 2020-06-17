
from data_load import load_vocab
from hyperparams import Hyperparams as hp
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf


class Graph:
    def __init__(self, num=1, mode="train"):
        '''
        Args:
          num: Either 1 or 2. 1 for Text2Mel 2 for SSRN.
          mode: Either "train" or "synthesize".
        '''
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Set flag
        training = True if mode=="train" else False

        # Graph
        # Data Feeding
        ## L: Text. (B, N), int32
        ## mels: Reduced melspectrogram. (B, T/r, n_mels) float32
        ## mags: Magnitude. (B, T, n_fft//2+1) float32

        self.L = tf.placeholder(tf.int32, shape=(None, None))
        self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
        self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))

        
        with tf.variable_scope("Text2Mel"):
            # Get S or decoder inputs. (B, T//r, n_mels)
            self.S = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

            # Networks
            with tf.variable_scope("TextEnc"):
                self.K, self.V = TextEnc(self.L, training=training)  # (N, Tx, e)

            with tf.variable_scope("AudioEnc"):
                self.Q = AudioEnc(self.S, training=training)

            with tf.variable_scope("Attention"):
                # R: (B, T/r, 2d)
                # alignments: (B, N, T/r)
                # max_attentions: (B,)
                self.R, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                            mononotic_attention=(not training),
                                                                            prev_max_attentions=self.prev_max_attentions)
            with tf.variable_scope("AudioDec"):
                self.Y_logits, self.Y = AudioDec(self.R, training=training) # (B, T/r, n_mels)
        
            # During inference, the predicted melspectrogram values are fed.
        with tf.variable_scope("SSRN"):
            self.Z_logits, self.Z = SSRN(self.Y, training=training)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        