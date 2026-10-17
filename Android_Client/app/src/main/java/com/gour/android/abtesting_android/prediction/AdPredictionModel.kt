import android.content.Context
import org.tensorflow.lite.Interpreter

import java.nio.ByteBuffer
import java.nio.ByteOrder

class AdPredictionModel(
    context: Context
) {

    private val interpreter: Interpreter

    init {
        val modelBytes =
            context.assets
                .open("ad_model_a.tflite")
                .readBytes()

        val modelBuffer =
            ByteBuffer.allocateDirect(modelBytes.size)
                .order(ByteOrder.nativeOrder())

        modelBuffer.put(modelBytes)
        modelBuffer.rewind()

        interpreter = Interpreter(modelBuffer)
    }

    fun predict(
        input: FloatArray
    ): Float {

        require(input.size == 8) {
            "Model A requires 8 features"
        }

        val modelInput = arrayOf(input)

        val output =
            Array(1) {
                FloatArray(1)
            }

        interpreter.run(
            modelInput,
            output
        )

        return output[0][0]
    }

    fun close() {
        interpreter.close()
    }
}