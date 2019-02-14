import uncertainties

def weightedaverage(values):
  values = tuple(values)
  if not values: raise IOError("Can't take the weighted average of an empty array")
  return uncertainties.ufloat(
    sum(x.nominal_value / x.std_dev**2 for x in values) / sum(1 / x.std_dev**2 for x in values),
    sum(1 / x.std_dev**2 for x in values) ** -0.5
  )

if __name__ == "__main__":
  print weightedaverage([uncertainties.ufloat(1, 1), uncertainties.ufloat(2, 2)])
