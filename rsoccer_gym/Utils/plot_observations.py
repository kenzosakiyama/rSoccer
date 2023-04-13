import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def compute_boundary_points_similarity(sigma=5, proportions=[]):
    # initial value
    likelihood_sample = 1

    # Loop over all observations for current particle
    for prop in proportions:
        # Map difference true and expected angle measurement to probability
        p_z_given_distance = \
            np.exp(-sigma * (1-prop) * (1-prop))

        # Incorporate likelihoods current landmark
        likelihood_sample *= p_z_given_distance
        if likelihood_sample<1e-15:
            return 0

    return likelihood_sample

def read_observations_log(path):
    quadrado_nr = []
    frame_nr = []
    measured_distance = []
    expected_distance = []
    angle = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print("Column names are ", ", ".join(row))
                line_count += 1
            else:
                quadrado_nr.append(int(row[0]))
                frame_nr.append(int(row[1]))
                measured_distance.append(float(row[2]))
                expected_distance.append(float(row[3]))
                angle.append(float(row[4]))
                line_count += 1
    
    return quadrado_nr, frame_nr, expected_distance, measured_distance, angle

def get_proportions_from_data(expected_distances, measured_distances):
    proportions = []
    for i in range(0, len(expected_distances)):
        prop = expected_distances[i]/measured_distances[i]
        if prop<2 and prop>0: 
            proportions.append(prop)
    
    return proportions

def get_errors_from_data(expected_distances, measured_distances):
    errors = []
    for (expected_distance, measured_distance) in zip(expected_distances, measured_distances):
        error = measured_distance - expected_distance
        errors.append(error)
        #if abs(error)<9: 
        #    errors.append(error)
        #else:
        #    errors.append(100)
    
    return errors

def get_percentual_errors_from_data(expected_distances, measured_distances):
    errors = []
    for (expected_distance, measured_distance) in zip(expected_distances, measured_distances):
        error = (measured_distance - expected_distance)/expected_distance
        errors.append(error)
        #if abs(error)<9: 
        #    errors.append(error)
        #else:
        #    errors.append(100)
    
    return errors

def set_histogram_plot(axis, errors, percentual=False):
    axis.set_title("Error Distribution")
    axis.set_ylabel("Probability")
    axis.set_xlabel("Error")

    xs = []
    for error in errors:
        if abs(error)<10: 
            xs.append(error)

    mn, mx = min(xs), max(xs)
    q25, q75 = np.percentile(xs, [25, 75])
    bin_width = 2 * (q75 - q25) * len(xs) ** (-1/3)
    bins = round((mx - mn) / bin_width)
    #if percentual:
    #    axis.hist(errors, density=True, bins=bins, label="Percentual")
    #else:
    #    axis.hist(errors, density=True, bins=bins, label="Absolute")
    print("Freedman-Diaconis number of bins:", bins)

    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(xs)
    if percentual:
        axis.plot(kde_xs, kde.pdf(kde_xs), label="Percentual")
    else:
        axis.plot(kde_xs, kde.pdf(kde_xs), label="Absolute")
    axis.legend(loc="upper left")

    xlim = max([abs(mn), abs(mx)])
    axis.set_xlim(-xlim, xlim+0.01)
    xticks = np.arange(-xlim, xlim+0.01, xlim/5)
    axis.set_xticks(xticks)
    axis.grid(True)

def set_error_plot(axis, errors, distances, percentual=False):
    axis.set_title("Error Size by Distance")
    axis.set_ylabel("Error")
    axis.set_xlabel("Measured Distance (m)")

    xs = []
    ys = []
    for error, distance in zip(errors, distances):
        if abs(error)<0.5:
            xs.append(distance)
            ys.append(error)

    if percentual:
        axis.scatter(xs, ys, label='Percentual', marker='D')
    else:
        axis.scatter(xs, ys, label='Absolute')
    axis.legend(loc="upper left")

if __name__ == "__main__":

    HISTOGRAM = True

    cwd = os.getcwd()
    path = cwd+f"/observations_data/log.csv"
    _, _, expected_distance, measured_distance, _ = read_observations_log(path=path)
    props = get_proportions_from_data(expected_distance, measured_distance)

    errors = get_errors_from_data(expected_distance, measured_distance)

    percentual_errors = get_percentual_errors_from_data(expected_distance, measured_distance)

    if HISTOGRAM:
        import scipy.stats as st

        fig = plt.figure(figsize=(10, 10))
        ax1, ax2 = fig.subplots(nrows=2, ncols=1)

        #set_histogram_plot(ax1, errors, False)
        set_histogram_plot(ax1, percentual_errors, True)
        #set_error_plot(ax2, errors, measured_distance, False)
        set_error_plot(ax2, percentual_errors, measured_distance, True)

    else:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.subplots(nrows=1, ncols=1)
        ax.set_title("Similarities Behavior")
        ax.set_xlabel("Particle to Robot Distance Proportion")
        ax.set_ylabel("Similarity")

        sigmas = []
        for sigma in range(0, 11, 1):
            xs = []
            ys = []
            if sigma==0:sigma=1
            sigmas.append(f'alpha = {sigma}')
            for prop in props:
                xs.append(prop)
                likelihood = compute_boundary_points_similarity(sigma=sigma,
                                                                proportions=[prop])
                ys.append(likelihood)
            ax.scatter(xs, ys)

        ax.legend(sigmas)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, max(xs))
        yticks = np.arange(0, 1.1, 0.1)
        ax.set_yticks(yticks)
        xticks = np.arange(0, 2.1, 0.1)
        ax.set_xticks(xticks)
        ax.grid(True)

    plt.show()
            
