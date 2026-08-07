package com.m2.inkrunner

import android.app.Activity
import android.os.Bundle
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognition
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognitionModel
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognitionModelIdentifier
import com.google.mlkit.vision.digitalink.recognition.DigitalInkRecognizerOptions
import com.google.mlkit.vision.digitalink.recognition.Ink
import com.google.mlkit.vision.digitalink.recognition.RecognitionContext
import com.google.mlkit.vision.digitalink.recognition.WritingArea
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Headless-ish ML Kit Digital Ink runner for the Mission-2 experiment.
 *
 * Reads synthetic stroke JSONs from the app-internal `files/in/` directory (populated via adb run-as),
 * recognizes each with the Hebrew (`he`) on-device model, and writes result
 * JSONs to `files/out/`. Driven entirely over adb; recognition itself is
 * fully on-device (network is used once, for the language-pack download).
 */
class MainActivity : Activity() {
    private val tag = "M2InkRunner"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread { runAll() }.start()
    }

    private fun runAll() {
        val base = filesDir
        val inDir = File(base, "in").apply { mkdirs() }
        val outDir = File(base, "out").apply { mkdirs() }
        val status = File(base, "status.txt")
        try {
            status.writeText("downloading-model")
            val modelId = DigitalInkRecognitionModelIdentifier.fromLanguageTag("he")
                ?: throw IllegalStateException("no model identifier for language tag 'he'")
            val model = DigitalInkRecognitionModel.builder(modelId).build()
            val manager = RemoteModelManager.getInstance()
            val t0 = System.currentTimeMillis()
            Tasks.await(manager.download(model, DownloadConditions.Builder().build()))
            val dlMs = System.currentTimeMillis() - t0
            Log.i(tag, "model ready in ${dlMs}ms")
            status.writeText("recognizing")

            val recognizer = DigitalInkRecognition.getClient(
                DigitalInkRecognizerOptions.builder(model).build()
            )
            val files = inDir.listFiles { f -> f.name.endsWith(".json") }?.sorted()
                ?: emptyList()
            for (f in files) {
                val outFile = File(outDir, f.name)
                if (outFile.exists()) continue
                val rec = JSONObject(f.readText())
                val result = JSONObject()
                result.put("item", rec.optString("item", f.nameWithoutExtension))
                result.put("model_download_ms", dlMs)
                try {
                    val inkBuilder = Ink.builder()
                    val strokes = rec.getJSONArray("strokes")
                    for (i in 0 until strokes.length()) {
                        val sb = Ink.Stroke.builder()
                        val pts = strokes.getJSONArray(i)
                        for (j in 0 until pts.length()) {
                            val p = pts.getJSONArray(j)
                            sb.addPoint(
                                Ink.Point.create(
                                    p.getInt(0).toFloat(), p.getInt(1).toFloat()
                                )
                            )
                        }
                        inkBuilder.addStroke(sb.build())
                    }
                    val ctx = RecognitionContext.builder()
                        .setPreContext("")
                        .setWritingArea(
                            WritingArea(
                                rec.getInt("width").toFloat(),
                                rec.getInt("height").toFloat()
                            )
                        )
                        .build()
                    val t1 = System.currentTimeMillis()
                    val res = Tasks.await(recognizer.recognize(inkBuilder.build(), ctx))
                    val ms = System.currentTimeMillis() - t1
                    val cands = JSONArray()
                    for (c in res.candidates) {
                        val cj = JSONObject()
                        cj.put("text", c.text)
                        c.score?.let { cj.put("score", it.toDouble()) }
                        cands.put(cj)
                    }
                    result.put("transcription",
                        if (res.candidates.isNotEmpty()) res.candidates[0].text else "")
                    result.put("candidates", cands)
                    result.put("recognition_ms", ms)
                    result.put("error", JSONObject.NULL)
                } catch (e: Exception) {
                    result.put("transcription", JSONObject.NULL)
                    result.put("error", "${e.javaClass.simpleName}: ${e.message}")
                }
                outFile.writeText(result.toString(2))
                Log.i(tag, "done ${f.name}")
            }
            status.writeText("done")
        } catch (e: Exception) {
            Log.e(tag, "fatal", e)
            status.writeText("fatal: ${e.javaClass.simpleName}: ${e.message}")
        }
    }
}
