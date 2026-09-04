import json
import requests

# ----------------------------
# CONFIG
# ----------------------------
OLLAMA_URL = "http://10.9.64.22:11434/api/generate"
MODEL_NAME = "deepseek-r1:32b"

# ----------------------------
# FUNCTION: Ask DeepSeek
# ----------------------------
def ask_deepseek(prompt: str):
    """
    Sends a query to DeepSeek model via Ollama and returns a readable response.
    """
    payload = {"model": MODEL_NAME, "prompt": prompt}
    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    if response.status_code != 200:
        raise Exception(f"❌ Failed request: {response.status_code} - {response.text}")

    full_output = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    full_output += data["response"]
            except json.JSONDecodeError:
                continue  # Skip malformed lines if any

    return full_output.strip()

# ----------------------------
# MAIN USAGE
# ----------------------------
if __name__ == "__main__":
    # ✏️ Paste your query here
    query = """
 4.3 Graph-based Structured Matching
The previous two components determine which user pairs
are worth considering for synchronization. RF–IMU distance estimation (§4.1) identifies users that are physically
close, while overlapping density prediction (§4.2) evaluates
whether their FoVs contain sufficient shared map points.
Together, these filters eliminate most unmatchable cases.
However, even for the remaining candidates, direct feature matching is inefficient because it requires exhaustive pairwise comparisons across thousands of points.
To address this, we represent each user’s map points as a
graph and perform structured matching. The graph formulation encodes both spatial proximity and local connectivity,
which allows us to prune the search space before featurelevel verification. As shown in Fig. 7, We derive spectral embeddings that preserve geometric structure [10, 28, 55, 57, 61]
before feature-level verification [45, 48].
Graph construction. For each user 𝑖, we construct a graph
𝐺𝑖 = (𝑉𝑖
, 𝐸𝑖), where each node 𝑣𝑗 ∈ 𝑉𝑖 represents a map
point. Nodes are annotated with local attributes: (1) ORB
descriptors, (2) RGB color values, and (3) node degree, which
reflects local connectivity. Edges are added between points
𝑗1 and 𝑗2 if
∏︁𝑃𝑗1 − 𝑃𝑗2
∏︁ < 𝑑max, (11)
with 𝑑max = 0.3 m, following common practice in co-visibility
graphs [11, 41]. This threshold balances local structure without introducing spurious long-range edges.
Spectral embeddings. We compute node-level spectral embeddings by constructing the graph Laplacian, performing
eigenvalue decomposition, and selecting the top-� eigenvectors. Each node is represented by a 𝑘-dimensional vector
𝜙𝑗 ∈ R
𝑘
. In our system, 𝑘 = 3, which preserves essential
graph structure while keeping representation compact for
transmission [10, 55].
Embedding exchange. When RF–IMU fusion and Bluetooth headers confirm that a new peer is within range, devices exchange only spectral embeddings rather than full
descriptors or maps. Each embedding is just three floatingpoint values, and further filtering based on distance and
orientation reduces them to likely overlapping FoV points.
This keeps bandwidth consumption minimal.
Triggered embedding updates. Embeddings are not recomputed continuously. Instead, they are refreshed only when
new peers are detected, and only for the subgraph around visible map points. On Jetson Orin Nano, each triggered update
completes within 20 ms, ensuring real-time responsiveness.
Spectral embedding-based matching. Given two graphs
𝐺𝐴 and 𝐺𝐵, embeddings Φ𝐴 ∈ R
𝑛𝐴×𝑘
and Φ𝐵 ∈ R
𝑛𝐵×𝑘
are
compared via dot-product similarity:
S = Φ𝐴 ⋅ Φ
𝑇
𝐵
. (12) Entries above a similarity threshold (0.75 in our implementation) are retained, with additional filtering to ensure neighborhood consistency (degree difference less than 10%).
Feature verification. This graph-based pruning reduces the
candidate set to a small fraction of points. For these, we apply
conventional ORB verification to finalize correspondences.
Since map points are derived from multi-view observations,
they remain stable across viewpoints. Combined with spectral embeddings that preserve neighborhood structure, this makes the system robust to orientation changes and significantly lowers computation.

explain the above in terms of a framework for decentralized synchronization for multi user XR applications.
    """

    print("🧠 Querying DeepSeek...\n")
    answer = ask_deepseek(query)

    print("✅ Readable Output:\n")
    print("=" * 80)
    print(answer)
    print("=" * 80)
