
ContrailBench
==============

*Benchmarking forecasts for contrail avoidance.*

========
Overview
========

Re-routing flights to avoid persistent contrails can
`significantly reduce aviation's climate impact <https://notebook.contrails.org/contrails-org-why-its-time-to-change-course-for-the-climate/>`_.
This is only possible with forecasts that support effective avoidance at a
`reasonable cost <https://notebook.contrails.org/the-cost-of-contrail-management/>`_.

**ContrailBench** is an open framework for evaluating contrail forecasts against
`real-world observations <https://notebook.contrails.org/observing-contrails-a-trifle-complicated/>`_
using metrics that capture forecast effectiveness and cost.
This site hosts reports evaluating `participating forecasts`_ alongside links
to the `ContrailBench evaluation code <https://github.com/contrailcirrus/contrail-bench>`_ (on Github) and observation-based `evaluation datasets`_ (on a public cloud bucket).

ContrailBench is an evolving framework. File a `GitHub issue`_
or write to info@contrails.org to share ideas and suggestions with the team.


=======
Reports
=======

Contrail forecasts can be evaluated using
`many <https://doi.org/10.3390/aerospace7120169>`_
`different <https://doi.org/10.1016/j.atmosres.2024.107663>`_
`metrics <https://doi.org/10.5194/acp-25-18051-2025>`_.
ContrailBench targets metrics that capture the tradeoff between
*effectiveness* (i.e., the fraction of persistent contrail kilometers or contrail radiative forcing
eliminated) and *cost* (i.e., the increase in operating expenses or CO2 emissions) when using a
particular forecast.
For a detailed primer on metrics used in ContrailBench reports, see our `notebook post <https://notebook.contrails.org/introducing-contrailbench>`_.

Static benchmark reports are released on a roughly quarterly cadence.
Frequent releases allow reports to add new forecasts
and evaluation datasets, expand observation coverage, and incorporate
improvements to evaluation methods.

All available reports are listed below.

.. table::
   :widths: auto

   ========== =====================
   **Report** **Coverage**
   ========== =====================
   `v1`_      January-December 2024
   ========== =====================

=======================
Participating forecasts
=======================

.. table::
   :widths: auto

   ================ ================================ =========
   **Forecast**     **Type**                         **Added**
   ================ ================================ =========
   `Contrails.org`_ Deterministic, physics-based     V1
   `Google`_        Probabilistic, ML-physics hybrid V1
   ================ ================================ =========

For examples showing how to access participating forecasts,
see `Participating Forecasts <forecasts.rst>`_.

Have a forecast you'd like to see included? Submit a `GitHub issue`_
or write to info@contrails.org and we'll work with you to include it in future reports.

===================
Evaluation datasets
===================

.. table::
   :widths: auto

   ================ ================================ =========
   **Observation**  **Type**                         **Added**
   ================ ================================ =========
   `ContrailWatch`_ Geostationary Satellite          V1
   `GRUAN`_         In-situ (radiosonde)             V1
   `IAGOS`_         In-situ (aircraft)               V1
   ================ ================================ =========

We also provide a public ADS-B dataset used to compute cost metrics.

For examples showing how to access and use evaluation datasets,
see `Evaluation Datasets <datasets.rst>`_.

Have an observational dataset you'd like to see included? Submit a `GitHub issue`_
or write to info@contrails.org and we'll assess its suitability.


.. toctree::
   :hidden:
   :caption: Reports

   reports/v1/v1.rst

.. toctree::
   :hidden:
   :caption: Examples

   Participating Forecasts <forecasts>
   Evaluation Datasets <datasets>

.. toctree::
   :hidden:
   :caption: Links

   Github <https://github.com/contrailcirrus/contrail-bench>
   Contrails.org <https://contrails.org>

.. _Github issue: https://github.com/contrailcirrus/contrail-bench/issues

.. _V1: reports/V1/V1.rst

.. _Contrails.org: https://apidocs.contrails.org/notebooks/forecast_api.html
.. _Google: https://developers.google.com/contrails/v1/forecast-description#ml-based_model

.. _GRUAN: https://www.gruan.org/
.. _IAGOS: https://www.iagos.org/
.. _ContrailWatch: https://developers.google.com/contrails/v1/ContrailWatch-description
