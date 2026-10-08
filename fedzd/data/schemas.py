"""Canonical schema definitions for the two datasets.

These encode the *real* column layouts of the public distributions so the loader
works unchanged on the real CSVs, and so the synthetic surrogate we validate with
here is schema-faithful.

References
----------
ToN_IoT "Processed Network dataset" (Zeek/Bro flow features). The full
`Network_dataset_*.csv` files carry 44 feature columns plus `label` (binary) and
`type` (attack family). Families: normal, backdoor, ddos, dos, injection, mitm,
password, ransomware, scanning, xss -- exactly the family set ZAID reports, which
is how we know they used the Network flow data (not the IoT telemetry sensors).

UNSW-NB15 partitioned CSVs (`UNSW_NB15_training-set.csv` / `..._testing-set.csv`),
45 columns including `attack_cat` and `label`.

ZAID preprocessing (arXiv 2602.16098): remove flow identifiers, IP addresses,
ports, timestamps and protocol identifiers, then RF+RFE -> Mutual Information,
keeping the top 40% of MI scores => 32 features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    # Columns dropped before modeling: identifiers / leakage (ZAID protocol).
    id_columns: List[str]
    # Non-identifier categorical features that survive to encoding.
    categorical_columns: List[str]
    label_column: str            # binary 0/1
    family_column: str           # attack family name (string)
    benign_family: str           # value of family_column meaning "no attack"
    # Map raw family strings -> normalized lowercase canonical family names.
    family_aliases: Dict[str, str] = field(default_factory=dict)
    # Canonical shared-feature space name -> this dataset's column, for the
    # cross-domain transfer test (Stage 5). Only semantically-aligned flow
    # features are included; documented + intentionally conservative.
    shared_feature_map: Dict[str, str] = field(default_factory=dict)

    def normalize_family(self, raw: str) -> str:
        r = str(raw).strip().lower()
        return self.family_aliases.get(r, r)


# --- ToN_IoT Network (primary) ---------------------------------------------

TONIOT_NETWORK_FEATURES = [
    # connection
    "proto", "service", "duration", "src_bytes", "dst_bytes", "conn_state",
    "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    # dns
    "dns_query", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD",
    "dns_RA", "dns_rejected",
    # ssl
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "ssl_subject",
    "ssl_issuer",
    # http
    "http_trans_depth", "http_method", "http_uri", "http_version",
    "http_request_body_len", "http_response_body_len", "http_status_code",
    "http_user_agent", "http_orig_mime_types", "http_resp_mime_types",
    # weird
    "weird_name", "weird_addl", "weird_notice",
]

TONIOT_SCHEMA = DatasetSchema(
    name="toniot_network",
    id_columns=["ts", "src_ip", "src_port", "dst_ip", "dst_port"],
    categorical_columns=[
        "proto", "service", "conn_state", "dns_query", "dns_AA", "dns_RD",
        "dns_RA", "dns_rejected", "ssl_version", "ssl_cipher", "ssl_resumed",
        "ssl_established", "ssl_subject", "ssl_issuer", "http_trans_depth",
        "http_method", "http_uri", "http_version", "http_user_agent",
        "http_orig_mime_types", "http_resp_mime_types", "weird_name",
        "weird_addl", "weird_notice",
    ],
    label_column="label",
    family_column="type",
    benign_family="normal",
    family_aliases={
        "ddos": "ddos", "dos": "dos", "mitm": "mitm", "man-in-the-middle": "mitm",
        "xss": "xss", "injection": "injection", "password": "password",
        "scanning": "scanning", "backdoor": "backdoor", "ransomware": "ransomware",
        "normal": "normal", "benign": "normal",
    },
    shared_feature_map={
        "duration": "duration",
        "src_bytes": "src_bytes",
        "dst_bytes": "dst_bytes",
        "src_pkts": "src_pkts",
        "dst_pkts": "dst_pkts",
    },
)

# --- UNSW-NB15 (cross-domain transfer) -------------------------------------

UNSW_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]

UNSW_SCHEMA = DatasetSchema(
    name="unsw_nb15",
    id_columns=["id"],
    categorical_columns=["proto", "service", "state"],
    label_column="label",
    family_column="attack_cat",
    benign_family="normal",
    family_aliases={
        "normal": "normal", "dos": "dos", "generic": "generic",
        "exploits": "exploits", "fuzzers": "fuzzers",
        "reconnaissance": "scanning",   # closest ToN_IoT analogue
        "analysis": "analysis", "backdoor": "backdoor",
        "backdoors": "backdoor", "shellcode": "shellcode", "worms": "worms",
        " ": "normal", "": "normal",
    },
    shared_feature_map={
        "duration": "dur",
        "src_bytes": "sbytes",
        "dst_bytes": "dbytes",
        "src_pkts": "spkts",
        "dst_pkts": "dpkts",
    },
)

SCHEMAS: Dict[str, DatasetSchema] = {
    TONIOT_SCHEMA.name: TONIOT_SCHEMA,
    UNSW_SCHEMA.name: UNSW_SCHEMA,
}

# Canonical shared-feature space used for cross-domain transfer (Stage 5).
SHARED_FEATURE_SPACE = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"]
