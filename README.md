# HDB Resale Price Intelligence 
### End-to-End Data Pipeline, Analytics Dashboard & Price Intelligence Platform for Singapore HDB Transactions
A data analytics and machine learning project that transforms raw HDB resale transaction data into an interactive business intelligence dashboard and price prediction system.

Built using Python, SQL-style data transformation workflows, Streamlit, geospatial analytics and machine learning.

## Executive Summary
Developed an end-to-end HDB resale intelligence platform using Python and data engineering workflows to analyse Singapore's public housing resale project.

The project ingests raw resale transaction data, performs automated data cleaning and feature engineering, and enriches the records with geospatial coordinates and nearest MRT/school distances and delivers an interactive dashboard for market trend analysis.

Key analysis outputs include:
- average resale price
- price per square metre by town
- location-based transaction mapping
- nearest amenity distance analysis
- price trend exploration across years and flat types

This solution helps uncover pricing drivers, location premiums and transaction patterns, enabling better data-driven property decision-making.

## Business Problem
Singapore's HDB resale market contains large volumes of transaction data, but raw datasets alone do not provide actionable insights.

Key business questions addressed:
- Which towns command the highest price per sqm premium?
- How does proximity to MRT stations and schools influence resale value?
- Which flat types show the strongest price appreciation?
- How do lease age and floor area affect pricing?
- Where are the potential undervalued resale opportunities?

The objective is to convert fragmented raw transaction data into a decision-support intelligence tool for:
- property market analysis
- pricing strategy
- investment opportunity identification
- urban planning insights

## Methodology
### 1. Data Extraction
Raw HDB resale transaction data was ingested from publicly available datasets.

Main datasets include:
- transaction month
- town
- flat type
- block/street
- floor area
- lease information
- resale price

### 2. Data Cleaning & Transformation
Using Python (Pandas + NumPy):
- cleaned missing/ inconsistent values
- standardised categorical fields
- converted data formats
- normalised town/block naming conventions
- validated numerical columns
Featured engineering performed:
- price_per_sqm
- remaining_lease_years
- storey_mid
- lease age
- year extracted from transaction month

### 3. Geospatial Feature Engineering
Addresses were geocoded into latitude and longitude coordinates.

Distance-based features calculated using the Haversine formula:
- distance to nearest MRT
- distance to nearest school

### 4. Exploratory Data Analysis
Built analytics to identify:
- town-level price differences
- floor area vs price relationships
- location clusters
- lease effect on pricing

Visualisations include:
- bar charts
- scatterplots
- KPI cards
- geospatial map layers

### 5. Dashboard Development
Built an interactive Streamlit dashboard with filters:
- years
- town
- flat type
- street/block
- floor area range

Dashboard components:
- KPI summary cards
- filtered transaction table
- downloadable CSV export
- price-per-sqm charts
- transaction location map
