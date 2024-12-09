import requests


def get_interpolations(host="http://127.0.0.1:5000", starting_idx=114, ending_idx=582, n_interps=20):
    response = requests.get(
        f"{host}/interpolations",
        params={"starting_idx": starting_idx, "ending_idx": ending_idx, "n_interps": n_interps}
    )
    data = response.json()

    print(f"2D interpolations element 0: {data['interpolations_2d'][0]}")
    #print(f"2D interpolations first element length: {len(data['interpolations_2d'][0])}")
    print(f"Images length: {len(data['interpolations_images'])}")
    print(f"Images first element length: {len(data['interpolations_images'][0])}")
    print(f"Base64 plot length: {len(data['plot'])}")

def get_points(host="http://127.0.0.1:5000"):
    response = requests.get(f"{host}/points")
    data = response.json()

    print(f"Points length: {len(data)}")
    print(data)


if __name__ == "__main__":
    #get_interpolations(starting_idx=114, ending_idx=582, n_interps=5)
    get_points()