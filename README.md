# ABTestingMLPlatform
End-to-end ML experimentation platform for building and deploying ML models, configuring A/B experiments through FastAPI and a web UI, running on-device inference, and collecting mobile exposure and conversion events for statistical experiment analysis.
## End-to-End ML Experimentation Flow

This project implements a production-inspired ML experimentation platform covering the complete lifecycle from **ML model development** to **experiment configuration, on-device inference, event collection, and statistical A/B analysis**.

### High-Level Architecture

```text
                         ┌─────────────────────────┐
                         │    Historical Data      │
                         │ Impressions / Clicks    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   ML Training Pipeline  │
                         │ Python + TensorFlow     │
                         └────────────┬────────────┘
                                      │
                         Train / Evaluate / Compare
                                      │
                         ┌────────────▼────────────┐
                         │      ML Models          │
                         │ Model A / Model B       │
                         │   .keras → .tflite      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Model Registry      │
                         │ Version / SHA-256 / URL │
                         └────────────┬────────────┘
                                      │
                                      │
             ┌────────────────────────┴────────────────────────┐
             │                                                 │
             ▼                                                 ▼
┌─────────────────────────┐                       ┌─────────────────────────┐
│ Experiment Management UI│                       │       FastAPI           │
│                         │──── Create/Update ───▶│ Experiment Config API   │
│ • Experiment name       │                       │                         │
│ • Model A / Model B     │                       │ • Experiment config     │
│ • Traffic allocation    │                       │ • Model metadata        │
│ • Metrics               │                       │ • Model download URL    │
│ • Start / Stop          │                       │ • Event ingestion       │
└─────────────────────────┘                       └────────────┬────────────┘
                                                              │
                                                    Fetch Configuration
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │      Android App        │
                                                 └────────────┬────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │   ExperimentManager     │
                                                 │                         │
                                                 │ Deterministic assignment│
                                                 │ User → Control/Treatment│
                                                 └────────────┬────────────┘
                                                              │
                                                    Select model version
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │      ModelManager       │
                                                 │                         │
                                                 │ • Download model        │
                                                 │ • SHA-256 verification  │
                                                 │ • Cache model           │
                                                 │ • Load LiteRT model     │
                                                 └────────────┬────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │   Candidate Ads         │
                                                 │                         │
                                                 │ Sports                  │
                                                 │ Technology              │
                                                 │ Travel                  │
                                                 │ Food                    │
                                                 └────────────┬────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │ On-Device ML Inference  │
                                                 │      LiteRT             │
                                                 │                         │
                                                 │ Predict P(click)        │
                                                 │ for every candidate     │
                                                 └────────────┬────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │      Ad Ranking         │
                                                 │ Sort by P(click) DESC   │
                                                 └────────────┬────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │    Display Top Ad       │
                                                 └────────────┬────────────┘
                                                              │
                                              ┌───────────────┴───────────────┐
                                              │                               │
                                              ▼                               ▼
                                    ┌──────────────────┐            ┌──────────────────┐
                                    │ Exposure Event   │            │  Click Event     │
                                    │                  │            │                  │
                                    │ experiment_id    │            │ experiment_id    │
                                    │ variant          │            │ variant          │
                                    │ model_version    │            │ ad_id            │
                                    │ user_id          │            │ user_id          │
                                    │ ad_id            │            │ timestamp         │
                                    └─────────┬────────┘            └────────┬─────────┘
                                              │                               │
                                              └───────────────┬───────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │ FastAPI Event Collector │
                                                 └────────────┬────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │   Experiment Dataset    │
                                                 │                         │
                                                 │ Exposures               │
                                                 │ Clicks / Conversions    │
                                                 │ Variant                 │
                                                 │ Model Version           │
                                                 └────────────┬────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │ Statistical Analysis    │
                                                 │                         │
                                                 │ • CTR                   │
                                                 │ • Absolute lift         │
                                                 │ • Relative lift         │
                                                 │ • Confidence interval   │
                                                 │ • p-value               │
                                                 │ • Statistical power     │
                                                 │ • MDE                   │
                                                 └────────────┬────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │ Experiment Decision     │
                                                 │                         │
                                                 │ Ship Model B            │
                                                 │ Keep Model A            │
                                                 │ or Continue Experiment  │
                                                 └─────────────────────────┘
```

## Experiment Lifecycle

The platform follows the complete ML experimentation lifecycle:

**1. Train Models**

Historical impression and click data is used to train multiple ranking models using **Python and TensorFlow**.

For example:

```text
Model A
8 features
AUC: 0.7045
Log Loss: 0.3858

Model B
11 features
AUC: 0.7189
Log Loss: 0.3810
```

Offline metrics determine whether a candidate model is promising enough to enter an online experiment.

---

**2. Convert Models for On-Device Inference**

Trained TensorFlow models are converted to **LiteRT/TFLite** format.

```text
TensorFlow Model
      ↓
.keras
      ↓
TFLite Conversion
      ↓
.tflite
      ↓
Android Device
```

This allows ranking predictions to execute directly on the mobile device with low latency.

---

**3. Register and Version Models**

Each deployable model is associated with metadata such as:

```text
model_id
model_version
download_url
sha256
feature_schema
created_at
```

The SHA-256 checksum allows the Android application to verify that a downloaded model has not been corrupted or modified.

---

**4. Create an Experiment**

An experiment is created using the **Experiment Management UI**.

Example:

```text
Experiment: ad-ranking-v2

Control:
    Model A

Treatment:
    Model B

Traffic:
    Control   50%
    Treatment 50%

Primary Metric:
    CTR

Minimum Detectable Effect:
    5%
```

The UI sends the configuration to the **FastAPI experiment service**.

---

**5. Fetch Experiment Configuration**

The Android application periodically retrieves the active experiment configuration.

Example:

```text
GET /experiments/active
```

The response contains information such as:

```text
experiment_id
variants
traffic_allocation
model_version
model_url
sha256
start_time
end_time
```

---

**6. Deterministic Variant Assignment**

`ExperimentManager` assigns a user to a variant using deterministic hashing.

```text
hash(userId + experimentId)
              ↓
           bucket
              ↓
      ┌───────┴───────┐
      ▼               ▼
   Control         Treatment
   Model A          Model B
```

The same user therefore remains in the same experiment group across application launches.

This prevents users from switching between models and contaminating experiment results.

---

**7. Dynamic Model Delivery**

`ModelManager` determines which model is required for the assigned variant.

If the model is not already available locally:

```text
Download model
      ↓
Verify SHA-256
      ↓
Cache model
      ↓
Load LiteRT Interpreter
```

This allows models to be updated without requiring a new Android application release.

---

**8. On-Device Ad Ranking**

The application receives several candidate ads.

```text
Sports
Technology
Travel
Food
```

The assigned ML model predicts the probability of a click for each candidate.

Example:

```text
Technology    0.61
Travel        0.48
Sports        0.35
Food          0.21
```

Candidates are sorted by predicted probability.

```text
Technology → Travel → Sports → Food
```

The highest-ranked candidate is displayed to the user.

---

**9. Exposure Logging**

An exposure event is recorded when the user is actually exposed to the experiment treatment.

Example:

```json
{
  "experiment_id": "ad-ranking-v2",
  "variant": "treatment",
  "model_version": "model-b-v1",
  "user_id": "user-123",
  "ad_id": "technology-101",
  "event": "exposure"
}
```

Logging exposure separately from assignment is important because a user may be assigned to an experiment but never actually see the experimental experience.

---

**10. Interaction and Conversion Events**

User actions are also recorded.

Examples include:

```text
ad_impression
ad_click
conversion
```

Events are sent from the Android application to the backend event ingestion API.

```text
Android
   ↓
Event API
   ↓
Event Storage
   ↓
Experiment Dataset
```

---

**11. Statistical A/B Analysis**

Experiment data is aggregated by variant.

The analysis calculates metrics such as:

```text
CTR
Absolute Lift
Relative Lift
Confidence Interval
p-value
Statistical Significance
Statistical Power
Minimum Detectable Effect
Sample Size
```

Example:

```text
Control CTR       10.2%
Treatment CTR     11.1%

Absolute Lift     +0.9 percentage points
Relative Lift     +8.8%

p-value           0.018
Confidence Level  95%
```

The analysis determines whether the observed improvement is likely caused by the new model rather than random variation.

---

**12. Experiment Decision**

Based on the experiment results, the system can recommend:

```text
Treatment statistically better
            ↓
       Ship Model B

No significant difference
            ↓
     Continue Experiment

Treatment performs worse
            ↓
       Keep Model A
```

The winning model can then become the new production baseline for the next experiment.

## Complete Lifecycle

```text
Data
 ↓
Model Training
 ↓
Offline Evaluation
 ↓
TFLite Conversion
 ↓
Model Registry
 ↓
Experiment Creation
 ↓
FastAPI Configuration
 ↓
Android Fetches Experiment
 ↓
Deterministic A/B Assignment
 ↓
Dynamic Model Download
 ↓
SHA-256 Verification
 ↓
On-Device ML Inference
 ↓
Candidate Ranking
 ↓
User Exposure
 ↓
Exposure / Click / Conversion Events
 ↓
Event Collection
 ↓
Experiment Dataset
 ↓
Statistical Analysis
 ↓
Experiment Decision
 ↓
Winning Model
 ↓
Next Experiment
```

This architecture demonstrates the complete lifecycle of a modern ML experimentation system—from **model development and deployment to online experimentation and data-driven model selection**.
