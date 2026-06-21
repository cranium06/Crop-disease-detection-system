# TEMPORARY — converts checkpoints/mobilenetv2_best.h5 to a TensorFlow.js
# LayersModel (web/model/model.json + one binary weight shard), without the
# tensorflowjs package (which is awkward to install on Windows). Writes the
# documented tfjs "layers-model" format: modelTopology = Keras JSON config,
# weights concatenated little-endian float32 in manifest order.
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import tensorflow as tf

OUT = "web/model"
os.makedirs(OUT, exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model("checkpoints/mobilenetv2_best.h5", compile=False)

topology = json.loads(model.to_json())

# Strip weight regularizers from the topology: they only add a loss term during
# training and are irrelevant for inference, but TensorFlow.js does not
# recognise Keras's "L2" regularizer class name and refuses to load the model.
def _strip_regularizers(o):
    if isinstance(o, dict):
        for k in list(o):
            if k.endswith("_regularizer"):
                o[k] = None
            else:
                _strip_regularizers(o[k])
    elif isinstance(o, list):
        for x in o:
            _strip_regularizers(x)

_strip_regularizers(topology)

entries, buffers = [], []
for w in model.weights:
    arr = np.asarray(w.numpy(), dtype=np.float32)
    name = w.name[:-2] if w.name.endswith(":0") else w.name
    entries.append({"name": name, "shape": list(arr.shape), "dtype": "float32"})
    buffers.append(arr.tobytes())

bin_path = os.path.join(OUT, "group1-shard1of1.bin")
with open(bin_path, "wb") as f:
    f.write(b"".join(buffers))

model_json = {
    "format": "layers-model",
    "generatedBy": "keras 2.12 (manual export)",
    "convertedBy": "custom h5->tfjs",
    "modelTopology": topology,
    "weightsManifest": [
        {"paths": ["group1-shard1of1.bin"], "weights": entries}
    ],
}
with open(os.path.join(OUT, "model.json"), "w") as f:
    json.dump(model_json, f)

total = sum(int(np.prod(e["shape"])) for e in entries)
print(f"Wrote {len(entries)} weight tensors ({total:,} params, "
      f"{os.path.getsize(bin_path)//1024} KB) to {OUT}/")
print("Input layer:", model.inputs[0].shape, "| Output:", model.outputs[0].shape)
