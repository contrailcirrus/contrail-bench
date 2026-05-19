
ContrailBench
==============

*Benchmarking forecasts for contrail avoidance.*

========
Overview
========

Re-routing flights to avoid forming persistent contrails and
`significantly reduce aviation's climate impact <https://notebook.contrails.org/contrails-org-why-its-time-to-change-course-for-the-climate/>`_
requires access to forecasts that support effective avoidance at a reasonable
`cost <https://notebook.contrails.org/the-cost-of-contrail-management/>`_.
ContrailBench is an open framework for evaluating forecasts for contrail avoidance using a
`wide range of contrail observations <https://notebook.contrails.org/observing-contrails-a-trifle-complicated/>`_.

This site contains reports evaluating `participating forecasts`_ using a standard set of metrics designed to capture their effectiveness and cost.
In addition, we provide access to the `ContrailBench evaluation code <https://github.com/contrailcirrus/contrail-bench>`_ through GitHub
and to observation-based `evaluation datasets`_ through a public cloud bucket.

ContrailBench is an evolving framework, and the community is welcome to file a `GitHub issue`_
or write to info@contrails.org to share ideas and suggestions with the ContrailBench team.

=======
Reports
=======

Contrail forecasts can be evaluated using
`many <https://doi.org/10.3390/aerospace7120169>`_
`different <https://doi.org/10.1016/j.atmosres.2024.107663>`_
`metrics <https://doi.org/10.5194/acp-25-18051-2025>`_.
ContrailBench is based on the premise that the most important
metrics for evaluating forecasts for contrail avoidance are those that capture the tradeoff between
*effectiveness* (i.e., the fraction of persistent contrail kilometers or contrail radiative forcing
eliminated) and *cost* (i.e., the increase in operating expenses or CO2 emissions) when using a
particular forecast.
For a detailed primer on metrics used in ContrailBench reports, see our `notebook post <#>`_.

Results from ContrailBench forecast benchmarking are released as static reports on a roughly
quarterly cadence. Frequent releases of updated reports allow benchmarks to add new forecasts
and evaluation datasets, expand spatial and temporal coverage as new observations become
available, and incorporate improvements to methods for evaluation.

All available reports are listed below.

.. table::
   :widths: auto

   ========== =====================
   **Report** **Coverage**
   ========== =====================
   `2026 Q1`_ January-December 2024
   ========== =====================

=======================
Participating forecasts
=======================

.. table::
   :widths: auto

   ================ ================================ =========
   **Forecast**     **Type**                         **Added**
   ================ ================================ =========
   `Contrails.org`_ Deterministic, physics-based     2026 Q1
   `Google`_        Probabilistic, ML-physics hybrid 2026 Q1
   ================ ================================ =========

For examples showing how to access participating forecasts, see `Participating Forecasts <forecasts.rst>`_.

Have a forecast you'd like to see included in ContrailBench? Please submit a `GitHub issue`_
or write to info@contrails.org and we'll work with you to include it in future reports.

===================
Evaluation datasets
===================

.. table::
   :widths: auto

   ================ ================================ =========
   **Observation**  **Type**                         **Added**
   ================ ================================ =========
   `ContrailWatch`_ Geostationary Satellite          2026 Q1
   `GRUAN`_         In-situ (radiosonde)             2026 Q1
   `IAGOS`_         In-situ (aircraft)               2026 Q1
   ================ ================================ =========

In addition, we provide a public ADS-B dataset used to compute metrics related to cost.

For examples showing how to access and use evaluation datasets, see `Evaluation Datasets <datasets.rst>`_.

Have an observational dataset you'd like to see used in ContrailBench?
Please submit a `GitHub issue`_ or write to info@contrails.org and we'll work with you
to determine whether it's suitable for ContrailBench.


.. toctree::
   :hidden:
   :caption: Reports

   reports/2026-Q1/2026-Q1.rst

.. toctree::
   :hidden:
   :caption: Examples

   Participating Forecasts <forecasts>
   Evaluation Datasets <datasets>

.. _Github issue: https://github.com/contrailcirrus/contrail-bench/issues

.. _2026 Q1: reports/2026-Q1/2026-Q1.rst

.. _Contrails.org: https://apidocs.contrails.org/notebooks/forecast_api.html
.. _Google: https://developers.google.com/contrails/v1/forecast-description#ml-based_model

.. _GRUAN: https://www.gruan.org/
.. _IAGOS: https://www.iagos.org/
.. _ContrailWatch: https://developers.google.com/contrails/v1/ContrailWatch-description
