# Split data
def split_X_y(data):
    try:
        data.set_index(["coid", "mdate"], inplace = True)
    except:
        pass
    finally:
        X = data.drop(["return", "Industry_Eng"], axis = 1)
        y = data["return"]
    return X, y