Projects
================

## Project Organization

Each of your projects will have a similar organization. There are
certain limitations on the size and type of files that can be pushed to
GitHub. There are also certain things that shouldn’t be accessible by
the public (e.g., data we have a license to access). For these reasons,
we have folders and files that are pushed to GitHub and those that are
not.

### Pushed to GitHub

  - `/Code` Each script should do something specific (like tidyverse
    functions), have a descriptive name, and include number prefixes if
    they are meant to be run in a certain order (e.g.,
    `01_import_data.R`, `02_clean_data.R`).
  - `/Data` While all data live here, only data that are small and can
    be shared publically will be pushed.
  - `/Figures` Figures live here, including images (saved as PNG files)
    and data referenced or used for tables, for use in the `README` and
    report.
  - `/Report` The report, without any PDF knits.
  - `README` This page, with any other organization details to make it
    easy to navigate your repository.

### Not Pushed to GitHub

  - `/Output` Output from model runs. These files tend to be too large.
    They are also something each user can create on their own.

## Project Workflow

  - Use RStudio projects.
  - Use `here::here()` to specify files paths.
  - Use tidyverse functions as much as possible.
  - Try and follow the [tidyverse style
    guide](https://style.tidyverse.org).
  - For the inference project, use [A Tidy Bayesian
    Workflow](tidy-bayesian-workflow.md) as a reference for iterative
    steps.

For general details on GitHub usage, project organization, and project
workflow, see [Using
Github](https://github.com/quant-seminar/seminar-materials/using-github.md).
