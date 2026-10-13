# AWS Setup — EMR + S3

> Walkthrough to reproduce the environment. **No credentials are stored here.**
> Replace every `<PLACEHOLDER>` with your own values and never commit real keys.

---

## 0. Prerequisites

- An AWS account with programmatic access (an IAM user, **not** the root account).
- An IAM access key pair stored only in `~/.aws/credentials` (never in code).
- `awscli` installed locally.

> ⚠️ **Security note**
> - Never store an access key in plain text, in a repo, or in the code.
> - Disable or delete access keys when you no longer need them.
> - Use least-privilege IAM permissions and rotate keys regularly.

---

## 1. Local environment

```bash
conda create -n p8 python=3.9
conda activate p8
conda install pandas pillow tensorflow pyspark pyarrow
```

## 2. AWS CLI

```bash
pip install --user awscli
which aws        # check install location

aws configure
# AWS Access Key ID     : <YOUR_ACCESS_KEY_ID>
# AWS Secret Access Key : <YOUR_SECRET_ACCESS_KEY>
# Default region name   : <YOUR_REGION>           e.g. eu-west-1
# Default output format  : json
```

Verify access:

```bash
aws s3 ls
```

## 3. S3 bucket

```bash
aws s3 mb s3://<YOUR_BUCKET_NAME> --region <YOUR_REGION>
# upload the Fruits-360 images
aws s3 cp ./data s3://<YOUR_BUCKET_NAME>/data --recursive
```

## 4. EMR cluster

- Launch an EMR cluster with **Spark** installed.
- Attach an EC2 key pair for SSH (`<YOUR_KEY>.pem`, kept private, never committed).
- Use a bootstrap action to install Python dependencies on the nodes
  (TensorFlow, Pillow, pandas, pyarrow). Use the provided
  [`scripts/bootstrap-emr.sh`](../scripts/bootstrap-emr.sh):

  ```bash
  aws s3 cp scripts/bootstrap-emr.sh s3://<YOUR_BUCKET>/bootstrap-emr.sh
  # EMR → Create cluster → Bootstrap actions → s3://<YOUR_BUCKET>/bootstrap-emr.sh
  ```
- Give the EMR role read/write access to your S3 bucket.

Connect to the cluster (SOCKS tunnel for the Spark/Jupyter UI):

```bash
ssh -i ~/.ssh/<YOUR_KEY>.pem -D 5555 hadoop@<EMR_MASTER_PUBLIC_DNS>
```

## 5. Run the notebook

Open `notebooks/p8_fruits_pyspark_aws.ipynb` on the cluster, set your bucket
path at the top, and run. Results are written back to S3 as Parquet.

---

<sub>No secrets in this file — placeholders only.</sub>
