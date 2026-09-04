import pickle


NoOfCUs = 2
NoOfDUs = 3

# Creating Topology
topology = {}

# topology[f"srscu{0}"] = [f"srsdu{0}"]
# topology[f"srscu{1}"] = [f"srsdu{1}", f"srsdu{2}"]
# topology[f"Host{0}"] = [f"srscu{0}", f"srscu{1}", f"srsdu{0}", f"srsdu{1}", f"srsdu{2}"]

topology = {
    "Host0": {
        "srscus": {
            "srscu0": {
                "connected_to": ["srsdu0"]
            },
            "srscu1": {
                "connected_to": ["srsdu1", "srsdu2"]
            }
        },
        "srsdus": ["srsdu0", "srsdu1", "srsdu2"]
    }
}


# # Form the graph where srscu0 connects to srsdu0, srscu1 to srsdu1, and so on
# for i in range(min(NoOfCUs, NoOfDUs)):  # Prevent index errors
#     topology[f"srscu{i}"] = [f"srsdu{i}"]

# UEsOfDUs = {}

# for i in range(NoOfDUs):
#     UEsOfDUs[f"srsdu{i}"] = []


# Display the graph
print(f'Topology is as follows: \n{topology}', end="\n\n")

pickle.dump(topology, open("topology.pkl", "wb"))