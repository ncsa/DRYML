from dryml.dry_repo import repo
import matplotlib.pyplot as plt


def box_and_whisker(repo: repo, func, func_args=None, func_kwargs=None,
                    selector_dict={}, fig_kwargs=None, **kwargs):
    if fig_kwargs is None:
        fig_kwargs = {}

    results = []
    labels = []

    for label in selector_dict:
        selector = selector_dict[label]
        scalar_results = repo.apply(
            func, func_args, func_kwargs,
            selector=selector,
            **kwargs)
        if scalar_results is None or len(scalar_results) == 0:
            print(f"WARNING: No models for label {label}, skipping.")
        else:
            results.append(scalar_results)
            labels.append(label)

    if len(results) == 0:
        print("ERROR! No models found for any label!")
        return

    # Create Figure
    fig_kwargs['figsize'] = fig_kwargs.get('figsize', (10, 10))
    fig, axes = plt.subplots(1, 1, **fig_kwargs)

    # Creating box plot
    axes.boxplot(results, labels=labels)

    # Show plots
    plt.show(fig)

    # Close Figure so we don't pollute the sessions with lots of open Figures
    plt.close(fig)
