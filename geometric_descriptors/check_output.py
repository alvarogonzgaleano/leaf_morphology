import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv("output/geometric_descriptors.csv")

print(df.describe().round(3).T)
print()
print("n_components != 1:", (df.n_components != 1).sum())          # should be 0
print("lw_ratio < 1:", (df.lw_ratio < 1).sum())                    # should be 0
print("circularity > 1.05:", (df.circularity > 1.05).sum())        # should be 0

# Relative difference between the two definitions of the length/width ratio
df["feret_ratio"] = df.max_feret_px / df.min_feret_px
df["rel_diff"] = (df.feret_ratio / df.lw_ratio - 1).abs()
print("\nrel_diff (|feret_ratio / lw_ratio - 1|):")
print(df.rel_diff.describe().round(3))
print((df.rel_diff > 0.1).sum(), "leaves with a difference > 10 %")
print(df.nlargest(10, "rel_diff")[["image", "lw_ratio", "feret_ratio", "circularity"]].round(3))

# Extremes: these are the images to inspect visually
print("\nHighest lw_ratio:")
print(df.nlargest(8, "lw_ratio")[["image", "lw_ratio", "circularity", "area_px"]].round(3))
print("\nLowest circularity:")
print(df.nsmallest(8, "circularity")[["image", "lw_ratio", "circularity", "perimeter_px"]].round(3))
print("\nLongest perimeter:")
print(df.nlargest(8, "perimeter_px")[["image", "circularity", "perimeter_px"]].round(3))