import { useState } from "react";

const BACKEND_URL = "https://branches-very-dana-correction.trycloudflare.com";

export default function MoviePosterModelComparison() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleFileChange(event) {
    const selectedFile = event.target.files[0];

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please upload an image file.");
      return;
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResults(null);
    setError("");
  }

  async function runModels() {
    if (!file) {
      setError("Please upload a poster image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("poster", file);

      const response = await fetch(`${BACKEND_URL}/api/predict-poster-genres`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Prediction request failed.");
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-neutral-950 px-6 py-10 text-white">
      <div className="mx-auto max-w-6xl">
        <h1 className="text-4xl font-bold">Movie Poster Genre Classifier</h1>
        <p className="mt-3 max-w-2xl text-neutral-400">
          Upload a movie poster and compare predictions from the Xenia, Tim, and Hannah models.
        </p>

        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-neutral-800 bg-neutral-900 p-6">
            <h2 className="text-xl font-semibold">Upload poster</h2>

            <input
              className="mt-5 block w-full rounded-lg border border-neutral-700 bg-neutral-950 p-3 text-sm"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />

            {previewUrl && (
              <div className="mt-6">
                <img
                  src={previewUrl}
                  alt="Poster preview"
                  className="max-h-[420px] rounded-xl border border-neutral-800 object-contain"
                />
                <p className="mt-3 text-sm text-neutral-400">{file?.name}</p>
              </div>
            )}

            <button
              onClick={runModels}
              disabled={loading || !file}
              className="mt-6 w-full rounded-xl bg-white px-4 py-3 font-semibold text-neutral-950 disabled:bg-neutral-700 disabled:text-neutral-400"
            >
              {loading ? "Running models..." : "Run all models"}
            </button>

            {error && (
              <div className="mt-4 rounded-xl border border-red-800 bg-red-950 p-4 text-sm text-red-200">
                {error}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-neutral-800 bg-neutral-900 p-6">
            <h2 className="text-xl font-semibold">Results</h2>

            {!results && !loading && (
              <p className="mt-5 text-neutral-400">
                Results will appear here after you run the models.
              </p>
            )}

            {loading && (
              <p className="mt-5 text-neutral-400">
                Running inference on the Colab backend...
              </p>
            )}

            {results && (
              <div className="mt-5 space-y-5">
                {results.models?.map((model) => (
                  <div
                    key={model.modelId}
                    className="rounded-xl border border-neutral-800 bg-neutral-950 p-5"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h3 className="text-lg font-semibold">{model.modelName}</h3>
                        <p className="text-xs text-neutral-500">{model.source}</p>
                      </div>

                      <div className="rounded-full border border-neutral-700 px-3 py-1 text-xs text-neutral-400">
                        {model.runtimeMs ? `${model.runtimeMs} ms` : "N/A"}
                      </div>
                    </div>

                    {model.error && (
                      <p className="mt-3 rounded-lg bg-red-950 p-3 text-sm text-red-200">
                        {model.error}
                      </p>
                    )}

                    <div className="mt-4">
                      <p className="text-sm font-medium text-neutral-300">
                        Predicted genres
                      </p>

                      <div className="mt-2 flex flex-wrap gap-2">
                        {model.predictedGenres?.length > 0 ? (
                          model.predictedGenres.map((genre) => (
                            <span
                              key={genre}
                              className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-neutral-950"
                            >
                              {genre}
                            </span>
                          ))
                        ) : (
                          <span className="text-sm text-neutral-500">
                            No genres above threshold.
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="mt-5">
                      <p className="text-sm font-medium text-neutral-300">
                        Top confidence scores
                      </p>

                      <div className="mt-3 space-y-2">
                        {model.topGenres?.map((item) => (
                          <div key={item.genre}>
                            <div className="mb-1 flex justify-between text-xs">
                              <span>{item.genre}</span>
                              <span>{Math.round(item.confidence * 100)}%</span>
                            </div>

                            <div className="h-2 rounded-full bg-neutral-800">
                              <div
                                className="h-2 rounded-full bg-white"
                                style={{
                                  width: `${Math.round(item.confidence * 100)}%`,
                                }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}

                {results.ensemble && (
                  <div className="rounded-xl border border-white bg-white p-5 text-neutral-950">
                    <h3 className="text-lg font-semibold">Combined ensemble view</h3>

                    <div className="mt-3 flex flex-wrap gap-2">
                      {results.ensemble.predictedGenres?.length > 0 ? (
                        results.ensemble.predictedGenres.map((genre) => (
                          <span
                            key={genre}
                            className="rounded-full bg-neutral-950 px-3 py-1 text-xs font-semibold text-white"
                          >
                            {genre}
                          </span>
                        ))
                      ) : (
                        <span className="text-sm text-neutral-600">
                          No genres predicted by multiple models.
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
