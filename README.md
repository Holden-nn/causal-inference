# IS 5150/6110 Advanced Regression for Causal Inference


**Class**: Monday and Wednesday 1:30-2:45 pm

- Room: HH 120

**Instructor**: Marc Dotson

- Contact: <marc.dotson@usu.edu>
- Office Hours: Monday and Wednesday 3-5 pm, or by appointment, EBB 705
  or
  [Zoom](https://usu-edu.zoom.us/j/9087876841?pwd=4Nl9sQnSAk3lXfwblJQduriCrzDYok.1)

**TA**: Holden Nielson

- Contact: <a02324537@usu.edu>
- Office Hours: By appointment

## Overview

This course focuses on the application of regression to inform
decision-making, particularly using interpretable models to understand
the effect of interventions on business outcomes. Students learn to
model experimental and observational data and infer causality instead of
correlation only.

By the end of this course, you will be able to:

1.  Specify identification strategies for estimating causal effects.
2.  Design effective experiments and apply appropriate methods for
    experimental data.
3.  Model observational data and infer causality using a variety of
    techniques.

## Tools

This course is heavily focused on skill-building. Each student will need
to bring a laptop, either their own or one rented from USU. All
assignments and project work will be completed using GitHub, an online
version control hub powered by Git. Walk through [this
training](https://github.com/marcdotson/asc-training) to get setup with
Python, Positron, Git and GitHub, and Quarto.

## *Probabilistic Machine Learning*

We will be studying Kevin Murphy’s excellent *Probabilistic Machine
Learning* series. Free PDFs of his books are available here:

- [*Probabilistic Machine Learning: An
  Introduction*](https://probml.github.io/pml-book/book1.html)
- [*Probabilistic Machine Learning: Advanced
  Topics*](https://probml.github.io/pml-book/book2.html)

Other free materials will be provided as needed.

## Studying

Students should consider the following study tips.

1.  Prepare for class by studying the assigned materials, taking notes,
    and coming with questions.
2.  Take notes, ask questions, and participate in class discussions.
3.  Consistently apply what you’re learning to your project.
4.  Work with classmates and utilize office hours.
5.  Use your class repository to organize all notes and project work.

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
| Participation | 50% |
| Project       | 50% |

No credit will be given for late work unless an arrangement is made
prior to the deadline. You are encouraged to review your graded work and
ask questions to avoid repeated mistakes.

### Participation

This class is all about participation. If you aren’t attending, you
can’t contribute. You will take turns preparing slides and presenting to
lead the discussion each class. When relevant, please include relevant
code when leading the discussion.

### Project

The project gives students the opportunity to demonstrate mastery over
the topics. Students will select what they want to study and work
iteratively on the project throughout the semester, applying techniques
and understanding as they acquire it. There will be an intermediate
presentation on the project halfway through through the semester and a
final presentation at the end of the semester. The project log of
milestones along with the final presentation slide deck will both be
part of your project grade.

## Schedule

Create a separate branch for each week’s work, merging to the `main`
branch and deleting the merged branch by the deadline **each Saturday
night**. Points will be lost for not following this workflow each week.

Please note that I reserve the right to change the syllabus, including
the schedule, at any time and for any reason. I will give advance notice
as it effects any deadlines.

#### Week 1: Workflow

- January 6: Python and Positron (Marc)
- January 8: GitHub and Quarto (Marc)
- *Probabilistic Machine Learning: An Introduction* Chapter 1
- Milestone 1: Draft Project Idea

#### Week 2: Probability

- January 13: Univariate Models (Rebecca)
- January 15: Multivariate Models (Marc)
- *Probabilistic Machine Learning: An Introduction* Chapters 2 and 3
- Milestone 2: Narrate the Data Story

#### Week 3: Causality

- January 20: No Class (MLK, Jr. Day)
- January 22: Structural Causal Models (Abby)
- *Probabilistic Machine Learning: Advanced Topics* Chapter 36.1-36.2
- Milestone 3: Specify a DAG

#### Week 4: Confounds and Controls

- January 27: [Elemental
  Confounds](https://www.youtube.com/watch?v=mBEA7PKDmiY&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=6)
  (Alesandro)
- January 29: [Good and Bad
  Controls](https://www.youtube.com/watch?v=uanZZLlzKHw&list=PLDcUM9US4XdPz-KxHM4XHt7uUVGWWVSus&index=7)
  (Jonah)
- *Probabilistic Machine Learning: Advanced Topics* Chapter 36.3-36.4
- Milestone 4: Specify an Identification Strategy

#### Week 5: Statistics

- February 3: Bayesian Statistics (Gabby)
- February 5: Frequentist Statistics (Rebecca)
- *Probabilistic Machine Learning: An Introduction* Chapter 4
- Milestone 5: Simulate Data and Recover Parameters

#### Week 6: Decision Theory

- February 10: Bayesian Decision Theory (Abby)
- February 12: Frequentist Decision Theory (Alesandro)
- *Probabilistic Machine Learning: An Introduction* Chapter 5
- Milestone 6: Conduct Exploratory Data Analysis

#### Week 7: PyMC

- February 17: No Class (President’s Day)
- February 19: Class Canceled

#### Week 8: PyMC and Presentations

- February 24: Introduction to PyMC (Marc)
- February 26: Intermediate Presentations
- Milestone 7: Estimate Causal Effects
- Milestone 8: Intermediate Presentation Slides

#### Week 9: Experimental Data

- March 3: Experimental Control
- March 5: Surveys, Conjoint, and Multilevel Models
- Milestone 9: Run Conjoint Experiment

#### Spring Break

- March 10: No Class (Spring Break)
- March 12: No Class (Spring Break)

#### Week 10: Observational Data

- March 17: Diff-in-Diff
- March 19: Diff-in-Diff
- Milestone 10: Implement Diff-in-Diff Strategy

#### Week 11: Synthetic Controls

- March 24: Matching
- March 26: Meta-Learning
- Milestone 11: Implement Synthetic Controls

#### Week 12: Causal Machine Learning

- March 31: Doubly Robust Machine Learning
- April 2: Other Methods
- Milestone 12: Implement Causal Machine Learning

#### Week 13: TBD

- April 7: TBD
- April 9: Marginal Effects
- Milestone 13: TBD

#### Week 14: Final Presentations

- April 14: Final Presentations
- April 16: Final Presentations
- Milestone 14: Final Presentations

#### Week 15: Next Steps

- April 21: What’s Next for Causal Inference?
- Milestone 15: Course Feedback

## HSB Differential Tuition

The Huntsman School of Business charges additional tuition, called
differential tuition, for 3000-level and above undergraduate business
school courses. More than 80 percent of this differential tuition is
used to recruit and retain the top-level faculty who teach in the
Huntsman School. We want our students to be informed about their
education and we welcome input from all of our stakeholders. The
Huntsman School Differential Tuition Advisory Board, comprised of
students, faculty, and staff, meets annually to review the uses of
differential tuition. More information about differential tuition is
online
at [https://huntsman.usu.edu/about/differential-tuitionLinks](https://huntsman.usu.edu/about/differential-tuition).

A few examples of expenses for which Differential Tuition is used
include but are not limited to:

- Salaries and benefits for Huntsman School faculty and staff
- New and existing student experiential programs
- Administrative infrastructure and operating expenses

## Assumption of Risk

All classes, programs, and extracurricular activities within the
University involve some risk, and certain ones involve travel. The
University provides opportunities to participate in these programs on a
voluntary basis. Therefore, students should not participate in them if
they do not care to assume the risks. Students can ask the respective
program leaders/sponsors about the possible risks a program may
generate, and if students are not willing to assume the risks, they
should not select that program. By voluntarily participating in classes,
programs, and extracurricular activities, a student does so at his or
her own risk. General information about University Risk Management
policies, insurance coverage, vehicle use policies, and risk management
forms can be found at: <http://www.usu.edu/riskmgt/>.

## Library Services

All USU students attending classes in Logan, at our Regional Campuses,
or online can access all databases, e-journals, and e-books regardless
of location. Additionally, the library will mail printed books to
students, at no charge to them. Students can also borrow books from any
Utah academic library. Take advantage of all library services and learn
more at [libguides.usu.edu/rc](http://libguides.usu.edu/rc).

## Classroom Civility

Utah State University supports the principle of freedom of expression
for both faculty and students. The University respects the rights of
faculty to teach and students to learn. Maintenance of these rights
requires classroom conditions that do not impede the learning process.
Disruptive classroom behavior will not be tolerated. An individual
engaging in such behavior may be subject to disciplinary action. Read
[Student Code Article V Section
V-3](https://studentconduct.usu.edu/studentcode/article5) for more
information.

## University Policies & Procedures

### Appropriate Use of Canvas and Other IT Resources

Canvas and all other course technologies are information technology
services provided as tools to further the mission of the university. By
using these services, users agree to comply with [USU Policy 550:
Appropriate Use of Computing, Networking, and Information
Resources](https://www.usu.edu/policies/550/) and the accompanying
[Terms of use for USU
IT](https://usu.service-now.com/aggies?id=kb_article_view&sysparm_article=KB0015388)
resources, as well as [Article
V-3.B.25.c](https://www.usu.edu/student-conduct/student-code/article5)
of the USU Student Code. Using course technologies in ways that are
inconsistent with the university’s mission or are disruptive will not be
tolerated. Disruptive behavior includes any activity that interferes
with either the faculty member’s ability to conduct the class or the
ability of other students to profit from the instructional program.

### Classroom Behavior

Utah State University supports the principle of freedom of expression
for both faculty and students. The University respects the rights of
faculty to teach and students to learn. Maintenance of these rights
requires classroom conditions that do not impede the learning process.
Disruptive classroom behavior will not be tolerated. An individual
engaging in such behavior may be subject to disciplinary action. Read
[Student Code Article V Section
V-3](https://www.usu.edu/student-conduct/student-code/article5) for more
information.

### Academic Freedom and Professional Responsibilities

Academic freedom is the right to teach, study, discuss, investigate,
discover, create, and publish freely. Academic freedom protects the
rights of faculty members in teaching and of students in learning.
Freedom in research is fundamental to the advancement of truth. Faculty
members are entitled to full freedom in teaching, research, and creative
activities, subject to the limitations imposed by professional
responsibility. [Faculty Code
Policy#403](https://www.usu.edu/policies/403/) further defines academic
freedom and professional responsibilities.

### Academic Integrity – “The Honor System”

Each student has the right and duty to pursue his or her academic
experience free of dishonesty. To enhance the learning environment at
Utah State University and to develop student academic integrity, each
student agrees to the following Honor Pledge:

*“I pledge, on my honor, to conduct myself with the foremost level of
academic integrity.”*

A student who lives by the Honor Pledge is a student who does more than
not cheat, falsify, or plagiarize. A student who lives by the Honor
Pledge:

- Espouses academic integrity as an underlying and essential principle
  of the Utah State University community;
- Understands that each act of academic dishonesty devalues every degree
  that is awarded by this institution; and
- Is a welcomed and valued member of Utah State University.

## Academic Dishonesty

The instructor of this course will take appropriate actions in response
to Academic Dishonesty, as defined the University’s Student Code.  Acts
of academic dishonesty include but are not limited to:

**Cheating:** using, attempting to use, or providing others with any
unauthorized assistance in taking quizzes, tests, examinations, or in
any other academic exercise or activity.  Unauthorized assistance
includes:

- Working in a group when the instructor has designated that the quiz,
  test, examination, or any other academic exercise or activity be done
  “individually;”
- Depending on the aid of sources beyond those authorized by the
  instructor in writing papers, preparing reports, solving problems, or
  carrying out other assignments;
- Substituting for another student, or permitting another student to
  substitute for oneself, in taking an examination or preparing academic
  work;
- Acquiring tests or other academic material belonging to a faculty
  member, staff member, or another student without express permission;
- Continuing to write after time has been called on a quiz, test,
  examination, or any other academic exercise or activity;
- Submitting substantially the same work for credit in more than one
  class, except with prior approval of the instructor; or engaging in
  any form of research fraud.

**Falsification:** altering or fabricating any information or citation
in an academic exercise or activity.

**Plagiarism:** representing, by paraphrase or direct quotation, the
published or unpublished work of another person as one‘s own in any
academic exercise or activity without full and clear acknowledgment. It
also includes using materials prepared by another person or by an agency
engaged in the sale of term papers or other academic materials.

For additional information go to: [ARTICLE VI. University Regulations
Regarding Academic
Integrity](https://www.usu.edu/student-conduct/student-code/article6).

### Discrimination and Sexual Misconduct

#### General Overview

USU strives to provide an environment for students and employees that is
free
from [discrimination](https://www.usu.edu/equity/non-discrimination) and [sexual
misconduct](https://www.usu.edu/equity/sexual-misconduct/Sexual-Misconduct-Terms).
If you experience sexual misconduct or discrimination at any point
during the semester inside or outside of class, you are encouraged to
contact the USU Title IX Coordinator via Distance Education room 400 in
Logan, 435-797-1266, <titleix@usu.edu>, or
at [equity.usu.edu/report](https://www.usu.edu/equity/report). You can
learn more about the USU resources available for individuals who have
experienced sexual misconduct
at [sexualrespect.usu.edu](https://www.usu.edu/sexual-respect/).
Resources for individuals who have experienced discrimination are listed
at [equity.usu.edu/resources](https://www.usu.edu/equity/resources).

#### Required Reporting of Sexual Misconduct and Threats of Harm

USU cares about our students and provides a number of resources and
supportive measures to students who may be experiencing thoughts of
self-harm or who have experienced sexual misconduct. To ensure students
are informed about resources and services available to them, including
available grievance or criminal processes for incidents of sexual
misconduct, USU has implemented [reporting policies and
practices](https://www.usu.edu/policies/340/) that require designated
employees to report any information they receive about incidents of
sexual misconduct. This reporting policy also assists USU with its
efforts to prevent sexual misconduct and keep our campus community
safe. 

Under USU’s sexual misconduct reporting policy, I am designated as a
[“reporting
employee.”](https://www.usu.edu/equity/sexual-misconduct/employees.php)
This means that if you share information with me about incidents of
[sexual
misconduct](https://www.usu.edu/equity/sexual-misconduct/Sexual-Misconduct-Terms.php) (sexual
harassment, sexual assault, relationship violence, or sex-based
stalking), including within a course assignment, I *will report* that
information to the [USU Title IX
Coordinator](https://www.usu.edu/equity/sexual-misconduct/Title-IX-Coordinator.php).
I will also share with you information about [designated confidential
resources](https://www.usu.edu/equity/sexual-misconduct/confidential-resources), [supportive
measures](https://www.usu.edu/equity/Supportive-Measures.php), and [how
you can file a report](https://www.usu.edu/equity/report.php) with the
USU Title IX Coordinator.

Self-disclosures about sexual misconduct that you experienced are not
required for your course work.

Similarly, if you disclose thoughts of harm to self or a threat to
others to me, including within a course assignment, I will report the
information to the appropriate campus administrators. I will also share
with you information about the [mental health and wellness
resources](https://www.usu.edu/aggiewellness/caps/) available to you. 

### Withdrawal Policy and “I” Grade Policy

Students are required to complete all courses for which they are
registered by the end of the semester. In some cases, a student may be
unable to complete all of the coursework because of extenuating
circumstances, but not due to poor performance or to retain financial
aid. The term ‘extenuating’ circumstances includes: (1) incapacitating
illness which prevents a student from attending classes for a minimum
period of two weeks, (2) a death in the immediate family, (3) financial
responsibilities requiring a student to alter a work schedule to secure
employment, (4) change in work schedule as required by an employer, or
(5) other emergencies deemed appropriate by the instructor.

### Students with Disabilities

USU welcomes students with disabilities. If you have, or suspect you may
have, a physical, mental health, or learning disability that may require
accommodations in this course, please contact the [Disability Resource
Center (DRC)](http://www.usu.edu/drc/) as early in the semester as
possible (University Inn \# 101, (435) 797‐2444,  <drc@usu.edu>). All
disability related accommodations must be approved by the DRC.  Once
approved, the DRC will coordinate with faculty to provide
accommodations.

### Students Who are Pregnant or Have a Pregnancy-Related Condition

If you need academic accommodations related to pregnancy, childbirth,
false pregnancy, termination of pregnancy, recovery, or other pregnancy
related conditions, please contact the Office of Equity as early as
possible. All accommodations related to pregnancy must be approved by
the Office of Equity. The Office of Equity will then coordinate with
instructors to provide accommodations.  The University will not exclude
a student from participating in any part of an educational program based
on the student’s pregnancy or pregnancy related conditions. 

*Office of Equity:* Distance Education, Room 400, Logan Campus,
435-797-1266, [Office of Equity: Pregnancy and Pregnancy Related
Conditions](https://www.usu.edu/equity/pregnancy-accommodations). 

### Inclusive Excellence

USU provides resources to help all students feel included as part of the
campus and broader USU community. To learn more about the resources
available and how to access them, visit the [Inclusive Excellence
Office](https://www.usu.edu/inclusive-excellence/).

### Grievance Process

Students who feel they have been unfairly treated may file a grievance
through the channels and procedures described in the [Academic
Grievances section of the Course
Catalog](https://catalog.usu.edu/content.php?catoid=39&navoid=30452).

### Full details for USU Academic Policies and Procedures

- [Acceptable Use of University Computing
  Resources](https://www.usu.edu/policies/550/)
- [Academic Policies and Practices (USU
  Catalog)](https://catalog.usu.edu/content.php?catoid=39&navoid=29998)
- [Student Conduct](http://www.usu.edu/studentconduct)
- [Student Code](https://www.usu.edu/student-conduct/student-code/)
- [Academic Freedom and Professional Responsibility
  Policy](https://www.usu.edu/policies/403/)

### Emergency Procedures

In the case of a drill or real emergency, classes will be notified to
evacuate the building via USU official communication channels.  Those
channels will be: an audible alarm, such as a fire alarm; an Aggie Alert
notification; or notification by a USU representative.  In the event of
a disaster that does not permit enough time for notifications, evacuate
as the situation dictates (i.e., when shaking ceases in an earthquake;
immediately when a fire is discovered or in the event of other immediate
life safety concerns). If it does not inhibit safety, turn off computers
and take any personal items with you. Elevators should not be used;
instead, use the closest stairs. See [USU Emergency
Management](https://www.usu.edu/dps/emergency/) for more information.

### General Health Protocols

The cold, flu, COVID-19, and other illnesses can have an impact on the
health of our university community. USU welcomes the wearing of masks in
all university buildings and encourages taking measures to mitigate risk
as recommended by federal and state public health officials: getting
vaccinated, staying home if you are sick, and frequent hand washing.

### Mental Health

Mental health is critically important for the success of USU students.
As a student, you may experience a range of issues that can cause
barriers to learning, such as strained relationships, increased anxiety,
alcohol/drug problems, feeling down, difficulty concentrating and/or
lack of motivation. These mental health concerns or stressful events may
lead to diminished academic performance or reduce your ability to
participate in daily activities. Utah State University provides free
services for students to assist them with addressing these and other
concerns. You can learn more about the broad range of confidential
mental health services available on campus at [Counseling and
Psychological Services (CAPS)](https://counseling.usu.edu/).

Students are also encouraged to download the “[SafeUT
App](https://safeut.org/)” to their smartphones. The SafeUT application
is a 24/7 statewide crisis text and tip service that provides real-time
crisis intervention to students through texting and a confidential tip
program that can help anyone with emotional crises, bullying,
relationship problems, mental health, or suicide related issues.

### Food Security

The Student Nutrition Access Center (SNAC) offers free food assistance
to all students. Students are welcome to visit the pantry once per
calendar week. There are no questions or qualifications required to
access this service; you simply need to present your student ID card or
A#.

The pantry is committed to supporting student well-being by ensuring
access to nutritious food options. For more information, including
pantry hours and location, please visit SNAC Food Pantry.

Utilizing the SNAC Food Pantry is a smart way to manage your food needs
and stay focused on your academic success.

**Disclaimer**: This syllabus is subject to change during the semester
based on the needs of the class.
