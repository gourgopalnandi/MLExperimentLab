package com.gour.android.abtesting_android

import AdPredictionModel
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.gour.android.abtesting_android.ui.theme.ABTesting_AndroidTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ABTesting_AndroidTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
        modelPrediction()
    }

    private fun modelPrediction(){
        val model = AdPredictionModel(this@MainActivity)

        val input = floatArrayOf(
            0.90f, // sports interest
            0.20f, // tech interest
            0.10f, // travel interest
            0.30f, // food interest

            1.0f,  // sports ad
            0.0f,  // tech ad
            0.0f,  // travel ad
            0.0f   // food ad
        )

        val prediction = model.predict(input)

        Log.d(
            "ML_TEST",
            "Click probability = ${prediction * 100}%"
        )
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}



@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    ABTesting_AndroidTheme {
        Greeting("Android")
    }
}