# IS 5150/6110 Advanced Regression for Causal Inference


**Class**: Monday and Wednesday 1:30-2:45 pm

- Room: HH 120

**Instructor**: Marc Dotson

- Contact: <marc.dotson@usu.edu>
- Office Hours: Monday and Wednesday 3-5 pm, or by appointment, EBB 705
  or
  [Zoom](https://usu-edu.zoom.us/j/9087876841?pwd=4Nl9sQnSAk3lXfwblJQduriCrzDYok.1)

## Overview

This course focuses on the application of regression to inform
decision-making, particularly using interpretable models to understand
the effect of interventions on business outcomes. Students learn to
model experimental and observational data and infer causality instead of
correlation only. Additional coursework is required for those enrolled
in the graduate-level course. Prerequisites/Restrictions: DATA 5600 Dual
listed as: IS 6110

By the end of this course, you will be able to:

1.  Specify identification strategies for estimating causal effects.
2.  Design effective experiments and apply appropriate methods for
    experimental data.
3.  Model observational data and infer causality using a variety of
    techniques.

## Tools and Materials

This course is heavily focused on skill-building. Analysis will be done
in R. R is a free, open-source statistical software package that is
commonly used in analytics. While it may take some getting used to,
fluency in R will provide students with a marketable and transferable
skill. We will be interfacing with a number of other languages from R to
accomplish specific modeling tasks.

Each student will need to bring a laptop, either their own or one rented
from BYU. All assignments and project work will be completed using
GitHub, an online version control hub powered by Git.

### R and RStudio

We will be using R via RStudio, which is a more coherent and visually
appealing environment for running R. Download and install the latest
version of [R](https://cran.r-project.org/) and the latest version of
[RStudio](https://posit.co/download/rstudio-desktop/), in that order.

### Git and GitHub

Sign up for a [GitHub](https://github.com/) account, join the [seminar
organization](https://github.com/quant-seminar), and create your seminar
repository `firstname-lastname` using the
[repo-template](https://github.com/quant-seminar/repo-template). Install
Git and GitHub working on your computer by completing Jenny Bryan’s
[pre-workshop
set-up](https://happygitwithr.com/workshops.html?mkt_tok=eyJpIjoiT1RVelptVTNZams0T0dZMiIsInQiOiJlR0orVlVpaHZsRlwveWh5QUJPN2U1Q3BcL0pHVHo5RXJ5UkhabFlwVXM4NlEwcHhRTENQZmVxaEEyNnVLSkRFTTdVa0hyNjk4MkFHYUU1Nkt5VXNtRm9heFM3N3dnUFplZ1V5anpRTWdnWDVscE1lOUR6VzBHaGFQOUFhOGd1QkN3In0=#pre-workshop-set-up).
This can take a few hours, so plan accordingly.

### Quarto

All of your work – assignments and project work – will be produced using
[Quarto](https://quarto.org/). This allows you to practice writing about
and working through problems as a report, explaining your concepts and
modeling choices in-line with visualizations and other output. If you
are familiar with R Markdown, Quarto is its successor.

### Stan

Stan is a probabilistic programming language that provides a modern
approach to doing Bayesian statistics. We will be interfacing with Stan
through CmdStanR. Carefully follow the [installation
instructions](https://mc-stan.org/cmdstanr/articles/cmdstanr.html). When
we get to using `ulam()` you’ll need to set the argument
`cmdstan = TRUE`.

### *Statistical Rethinking*

We will be using the second edition of Richard McElreath’s *Statistical
Rethinking*, including his [recorded
lectures](https://github.com/rmcelreath/stat_rethinking_2023). After
installing CmdStanR, install the McElreath’s accompanying {rethinking}
package from GitHub using the following:

    install.packages(c("coda","mvtnorm","devtools","loo","dagitty","shape"))
    devtools::install_github("rmcelreath/rethinking")

While *Statistical Rethinking* is written using base R, there are
translations, including one using {ggplot2} and other tidyverse packages
[here](https://bookdown.org/content/4857/).

### Supplementary Materials

There are three supplementary books that students may find useful.

- The first book is *R for Data Science (2e)* by Hadley Wickham and
  Garrett Grolemund, which is available for free online at
  [r4ds.hadley.nz](https://r4ds.hadley.nz). This book is particularly
  helpful for learning more about the
  [tidyverse](https://www.tidyverse.org).
- The second book is *Tidy Modeling with R* by Max Kuhn and Julia Silge,
  which is available for free online at
  [tmwr.org](https://www.tmwr.org). This book is particularly helpful
  for learning more about [tidymodels](https://www.tidymodels.org).
- The third book is *Causal Inference: The Mixtape* by Scott Cunningham,
  which is available for free online at
  [mixtape.scunning.com](https://mixtape.scunning.com). This book
  provides an overview of causal inference methods, including DAGs.

### *The Internet*

Students will inevitably end up searching for help online, though it is
highly recommended that they search slides, notes, and the supplementary
material first. When that fails, the [Posit
Community](https://community.rstudio.com), the [Stan
Forums](https://discourse.mc-stan.org), and [Stack
Overflow](http://stackoverflow.com), will be most helpful to push
through inevitable roadblocks.

### *GitHub Copilot*

Students can sign up for [GitHub
Copilot](https://github.com/features/copilot) as a tool to improve their
productivity. RStudio integration is enabled under
`Global Options > Copilot`.

## Studying

A seminar is different from other courses students have taken. Students
should consider the following study tips.

1.  Seek learning by study and faith (D&C 109:7).
2.  Prepare for the seminar by watching the lecture, working through
    assigned chapters/sections, taking notes, working on assignments,
    and coming with questions.
3.  Take notes, ask questions, and participate in seminar discussions.
4.  Consistently apply what you’re learning to your project work.
5.  Work with classmates and utilize office hours.
6.  Use your seminar repository to organize all notes, assignments, and
    project work.

## Assessment

Letter grades will follow the standard rubric.

|     |         |     |        |     |        |
|:----|:--------|:----|:-------|:----|:-------|
| A   | 93-100% | B-  | 80-82% | D+  | 67-69% |
| A-  | 90-92%  | C+  | 77-79% | D   | 63-66% |
| B+  | 87-89%  | C   | 73-76% | D-  | 60-62% |
| B   | 83-86%  | C-  | 70-72% | E   | 0-59%  |

Grades will be determined as follows.

|               |     |
|:--------------|:---:|
| Presentations | 20% |
| Assignments   | 30% |
| Project       | 50% |

No credit will be given for late work unless an arrangement is made
prior to the deadline. You are encouraged to review your graded
assignments and the solutions to avoid repeated mistakes.

### Presentations

A seminar is all about participation. If you aren’t attending, you can’t
contribute. We will take turns preparing slides and presenting to lead
the discussion each session using the following schedule:

- On Tuesdays, review the completed assignment and then discuss that
  day’s topic.
- On Thursdays, discuss that day’s topic and then work on the current
  assignment.

Please note that I will lead the review of the completed assignment and
work on the current assignment. However, as part of your presentation
when assigned to lead the discussion for the day’s topic, please include
code walkthroughs.

### Assignments

All of your assignments will be submitted on your GitHub repository.
Please do your own work and avoid looking at solutions until after the
assignment is due. You are welcome to ask questions and work with
classmates, but your work should be your own.

### Project

The project gives students the opportunity to demonstrate mastery over
the topics. Students will select what they [want to
study](https://github.com/quant-seminar/seminar-materials/blob/main/goldfarb-seminar-2022.pdf)
and work iteratively on the project throughout the semester, applying
techniques and understanding as they acquire it. There will be an
intermediate presentation on the project halfway through through the
semester and a final presentation at the end of the semester. The
project log of milestones along with the final presentation slide deck
will both be part of your project grade.

## Schedule

Create a separate branch for each week’s work, merging to the main
branch and deleting the merged branch by the deadline **each Saturday
night**. Points will be lost for not following this workflow each week.

Please note that I reserve the right to change the syllabus, including
the schedule, at any time and for any reason. I will give advance notice
as it effects any deadlines.

#### Week 1: Workflow

- January 9: Syllabus and GitHub
  ([Notes](https://github.com/marcdotson/ra-training#using-github))
- January 11: R/RStudio and Quarto
  ([Slides](https://github.com/quant-seminar/seminar-materials/blob/main/Presentations/R:RStudio%20and%20Quarto.qmd),
  [Notes](https://github.com/marcdotson/ra-training#using-quarto))
- *R for Data Science (2e)* Chapters 3, 4, 5, and 30
- Milestone 1: Draft an Abstract

#### Week 2: Quantitative Marketing

- January 16: The Tidyverse and Base R
  ([Slides](https://github.com/quant-seminar/seminar-materials/blob/main/Presentations/The%20Tidyverse%20and%20Base%20R.qmd))
- January 18: Paper Presentations, Getting a PhD
- *R for Data Science (2e)* Chapters 2, 6, 7, and 29
- Paper Presentation, Milestone 2: Narrate the Data Story

#### Week 3: Bayesian Inference

- January 23: The Golem of Prague
  ([Lecture](https://www.youtube.com/watch?v=FdnMWdICdRs&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=1),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-01))
- January 25: The Garden of Forking Data
  ([Lecture](https://www.youtube.com/watch?v=R1vcdhPBlXA&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=3),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-02))
- *Statistical Rethinking* Chapters 1, 2, and 3
- [Assignment
  1](https://github.com/quant-seminar/seminar-materials/blob/main/Assignments/assignment_01.md),
  Milestone 3: Simulate Data and Recover Parameters

#### Week 4: Linear Models and Causal Inference

- January 30: Geocentric Models
  ([Lecture](https://www.youtube.com/watch?v=tNOu-SEacNU&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=4),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-03))
- February 1: Categories and Curves
  ([Lecture](https://www.youtube.com/watch?v=F0N4b7K_iYQ&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=5),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-04))
- *Statistical Rethinking* Chapter 4
- [Assignment
  2](https://github.com/quant-seminar/seminar-materials/blob/main/Assignments/assignment_02.md),
  Milestone 4: Summarize the Data and Fit a Bayesian Regression,
  Including Prior and Posterior Predictive Plots

#### Week 5: Causes, Confounds, and Colliders

- February 6: Elemental Confounds
  ([Lecture](https://www.youtube.com/watch?v=mBEA7PKDmiY&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=6),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-05))
- February 8: Good and Bad Controls
  ([Lecture](https://www.youtube.com/watch?v=uanZZLlzKHw&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=7),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-06))
- *Statistical Rethinking* Chapters 5 and 6
- [Assignment
  3](https://github.com/quant-seminar/seminar-materials/blob/main/Assignments/assignment_03.md),
  Milestone 5: Specify a DAG and Fit a Second Model, Including a
  Counterfactual Plot

#### Week 6: Regularization and MCMC

- February 13: Fitting Over and Under
  ([Lecture](https://www.youtube.com/watch?v=1VgYIsANQck&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=8),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-07))
- February 15: Markov Chain Monte Carlo
  ([Lecture](https://www.youtube.com/watch?v=rZk2FqX2XnY&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=9),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-08))
- *Statistical Rethinking* Chapters 7 and 9
- [Assignment
  4](https://github.com/quant-seminar/seminar-materials/blob/main/Assignments/assignment_04.md),
  Milestone 6: Fit a Third Model Using MCMC, Including Model Comparisons

#### Week 7: Stan

- February 22: Stan
  ([Slides](https://github.com/quant-seminar/seminar-materials/blob/main/Presentations/Stan.qmd))
- Milestone 7: Translate Model into Stan and Use {bayesplot}

#### Week 8: Intermediate Presentations

- February 27: Discuss Projects
- February 29: Present Projects
- Intermediate Presentation, Milestone 8: Iterate Through the Entire
  Workflow

#### Week 9: Generalized Linear Models

- March 5: Modeling Events
  ([Lecture](https://www.youtube.com/watch?v=Zi6N3GLUJmw&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=10),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-09))
- March 7: Counts and Confounds
  ([Lecture](https://www.youtube.com/watch?v=jokxu18egu0&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=11),
  [Slides](https://speakerdeck.com/rmcelreath/statistical-rethinking-2023-lecture-10))
- *Statistical Rethinking* Chapters 10 and 11
- [Assignment
  5](https://github.com/quant-seminar/seminar-materials/blob/main/Assignments/assignment_05.md),
  Milestone 9: Add Complexity to the Model, Using a GLM as Needed

#### Week 10: Multilevel Models

- March 12: Multilevel Models
  ([Lecture](https://www.youtube.com/watch?v=iwVqiiXYeC4&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=13),
  [Slides](https://raw.githubusercontent.com/rmcelreath/stat_rethinking_2023/main/slides/Lecture_12-GLMM1.pdf))
- March 14: Multilevel Adventures
  ([Lecture](https://www.youtube.com/watch?v=sgqMkZeslxA&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=14),
  [Slides](https://raw.githubusercontent.com/rmcelreath/stat_rethinking_2023/main/slides/Lecture_13-GLMM2.pdf))
- *Statistical Rethinking* Chapter 13
- [Assignment
  6](https://github.com/quant-seminar/seminar-materials/blob/main/Assignments/assignment_06.md),
  Milestone 10: Specify a Multilevel Model with Varying Effects

#### Week 11: Multivariate Adaptive Priors

- March 19: Correlated Features
  ([Lecture](https://www.youtube.com/watch?v=Es44-Bp1aKo&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=15),
  [Slides](https://github.com/rmcelreath/stat_rethinking_2023/raw/main/slides/Lecture_14-GLMM3.pdf))
- March 21: Model Parameterizations
  ([Slides](https://github.com/quant-seminar/seminar-materials/blob/main/Presentations/Model%20Parameterizations.qmd))
- *Statistical Rethinking* Chapter 14.1-14.2
- [Assignment
  7](https://github.com/quant-seminar/seminar-materials/blob/main/Assignments/assignment_07.md),
  Milestone 11: Specify a Non-Centered Multilevel Model with Correlated
  Varying Effects

#### Week 12: Measurement and Missingness

- March 26: Measurement and Misclassification
  ([Lecture](https://www.youtube.com/watch?v=mt9WKbQJrI4&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=18),
  [Slides](https://github.com/rmcelreath/stat_rethinking_2023/raw/main/slides/Lecture_17-measurement.pdf))
- March 28: Missing Data
  ([Lecture](https://www.youtube.com/watch?v=Oeq6GChHOzc&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=19),
  [Slides](https://github.com/rmcelreath/stat_rethinking_2023/raw/main/slides/Lecture_18-missing_data.pdf))
- *Statistical Rethinking* Chapter 15
- ~~Assignment 8~~, Milestone 12: Finalize the Multilevel Model

#### Week 13: Generalized Linear Madness

- April 2: Generalized Linear Madness
  ([Lecture](https://www.youtube.com/watch?v=zffwg0xDOgE&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=20),
  [Slides](https://github.com/rmcelreath/stat_rethinking_2023/raw/main/slides/Lecture_19-GenLinearMadness.pdf))
- April 4: Horoscopes
  ([Lecture](https://www.youtube.com/watch?v=qwF-st2NGTU&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=21),
  [Slides](https://github.com/rmcelreath/stat_rethinking_2023/raw/main/slides/Lecture_20-horoscopes.pdf))
- *Statistical Rethinking* Chapter 16
- [Student Ratings](https://studentratings.byu.edu), Milestone 13: Draft
  a Working Paper

#### Week 14: Final Presentations

- April 9: Present Projects
- April 11: Present Projects
- Milestone 14: Revise the Working Paper

#### Week 15: Final Presentations

- April 16: Present Projects
- Final Presentation, Milestone 15: Finalize the Working Paper

## Honor Code

In keeping with the principles of the BYU Honor Code, students are
expected to be honest in all of their academic work. Academic honesty
means, most fundamentally, that any work you present as your own must in
fact be your own work and not that of another. Violations of this
principle may result in a failing grade in the course and additional
disciplinary action by the university. Students are also expected to
adhere to the Dress and Grooming Standards. Adherence demonstrates
respect for yourself and others and ensures an effective learning and
working environment. It is the university’s expectation, and every
instructor’s expectation in class, that each student will abide by all
Honor Code standards. Please call the Honor Code Office at 422-2847 if
you have questions about those standards.

## Preventing & Responding to Sexual Misconduct

Brigham Young University prohibits all forms of sexual harassment –
including sexual assault, dating violence, domestic violence, and
stalking on the basis of sex – by its personnel and students and in all
its education programs or activities. University policy requires all
faculty members to promptly report incidents of sexual harassment that
come to their attention in any way and encourages reports by students
who experience or become aware of sexual harassment. Incidents should be
reported to the Title IX Coordinator at <t9coordinator@byu.edu> or (801)
422-8692 or 1085 WSC. Reports may also be submitted online at
[titleix.byu.edu/report](https://titleix.byu.edu/report) or
1-888-238-1062 (24-hours a day). BYU offers a number of resources and
services for those affected by sexual harassment, including the
university’s confidential Sexual Assault Survivor Advocate. Additional
information about sexual harassment, the university’s Sexual Harassment
Policy, reporting requirements, and resources can be found in the
University Catalog, by visiting
[titleix.byu.edu](http://titleix.byu.edu), or by contacting the
university’s Title IX Coordinator.

## Student Disability

Brigham Young University is committed to providing a working and
learning atmosphere that reasonably accommodates qualified persons with
disabilities. A disability is a physical or mental impairment that
substantially limits one or more major life activities. Whether an
impairment is substantially limiting depends on its nature and severity,
its duration or expected duration, and its permanent or expected
permanent or long-term impact. Examples include vision or hearing
impairments, physical disabilities, chronic illnesses, emotional
disorders (e.g., depression, anxiety), learning disorders, and attention
disorders (e.g., ADHD). If you have a disability which impairs your
ability to complete this course successfully, please contact the
University Accessibility Center (UAC), 2170 WSC or 801-422-2767 to
request a reasonable accommodation. The UAC can also assess students for
learning, attention, and emotional concerns. If you feel you have been
unlawfully discriminated against on the basis of disability, please
contact the Equal Employment Office at 801-422-5895, D-285 ASB for help.

## Inappropriate Use of Course Materials

All course materials (e.g., outlines, handouts, syllabi, exams, quizzes,
PowerPoint presentations, lectures, audio and video recordings, etc.)
are proprietary. Students are prohibited from posting or selling any
such course materials without the express written permission of the
professor teaching this course. To do so is a violation of the Brigham
Young University Honor Code.

## Academic Honesty

The first injunction of the Honor Code is the call to “be honest.”
Students come to the university not only to improve their minds, gain
knowledge, and develop skills that will assist them in their life’s
work, but also to build character. “President David O. McKay taught that
character is the highest aim of education” (The Aims of a BYU Education,
p.6). It is the purpose of the BYU Academic Honesty Policy to assist in
fulfilling that aim. BYU students should seek to be totally honest in
their dealings with others. They should complete their own work and be
evaluated based upon that work. They should avoid academic dishonesty
and misconduct in all its forms, including but not limited to
plagiarism, fabrication or falsification, cheating, and other academic
misconduct.

## Marriott School of Business Inclusion Statement

At Brigham Young University’s Marriott School of Business, we embrace
the university’s mission to “assist individuals in their quest for
perfection and eternal life.” We strive to foster an environment that is
respectful of all backgrounds, perspectives, and voices, that “all may
be edified of all” (D&C 88:122). By extending a spirit of consideration,
fellowship, and charity to everyone, we enable the discovery of common
values and unique insights as we each pursue our worthy secular and
spiritual goals.

We embrace the statement President Russell M. Nelson made on June 1,
2020.

*“The Creator of us all calls on each of us to abandon attitudes of
prejudice against any group of God’s children. Any of us who has
prejudice toward another race needs to repent!*

*During the Savior’s earthly mission, He constantly ministered to those
who were excluded, marginalized, judged, overlooked, abused, and
discounted. As His followers, can we do anything less?*

*Let us be clear. We are brothers and sisters, each of us the child of a
loving Father in Heaven. His Son, the Lord Jesus Christ, invites all to
come unto Him – ‘black and white, bond and free, male and female,’ (2
Nephi 26:33). It behooves each of us to do whatever we can in our
spheres of influence to preserve the dignity and respect every son and
daughter of God deserves.”*

## Mental Health

Mental health concerns and stressful life events can affect students’
academic performance and quality of life. BYU Counseling and
Psychological Services (CAPS, 1500 WSC, 801-422-3035, caps.byu.edu)
provides individual, couples, and group counseling, as well as stress
management services. These services are confidential and are provided by
the university at no cost for full-time students. For general
information please visit [caps.byu.edu](https://caps.byu.edu); for more
immediate concerns please visit [help.byu.edu](http://help.byu.edu).
