import tensorflow as tf

def main():
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('model.tflite', 'wb') as f:
      f.write(tflite_model)

if __name__ == '__main__':
    main()
