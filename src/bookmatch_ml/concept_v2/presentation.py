"""Display labels for stable concept IDs; never derive graph edges from chapter order."""

from bookmatch_ml.config import LoadedRankingV2Config

LABELS = {
    "high school algebra": "고교 대수",
    "systems of equations": "연립방정식",
    "linear system": "선형 연립방정식",
    "gaussian elimination": "가우스 소거법",
    "matrix": "행렬",
    "vector": "벡터",
    "vector space": "벡터 공간",
    "linear independence": "선형 독립",
    "basis": "기저",
    "dimension": "차원",
    "rank": "행렬의 계수",
    "determinant": "행렬식",
    "eigenvalue": "고유값",
    "eigenvector": "고유벡터",
    "orthogonality": "직교성",
    "inner product": "내적",
    "linear transformation": "선형변환",
    "diagonalization": "대각화",
    "least squares": "최소제곱법",
    "singular value decomposition": "특이값 분해",
    "computer architecture": "컴퓨터 구조",
    "programming": "프로그래밍",
    "process": "프로세스",
    "thread": "스레드",
    "scheduling": "스케줄링",
    "concurrency": "동시성",
    "synchronization": "동기화",
    "deadlock": "교착상태",
    "memory management": "메모리 관리",
    "virtual memory": "가상 메모리",
    "storage": "저장장치",
    "file system": "파일 시스템",
    "input/output": "입출력",
    "protection": "보호",
    "security": "보안",
    "virtualization": "가상화",
    "distributed systems": "분산 시스템",
}


def concept_graph(config: LoadedRankingV2Config, topic: str) -> dict[str, object]:
    policy = config.config
    if topic not in policy.nodes:
        raise ValueError("this topic has no reviewed concept graph")
    return {
        "topicId": topic,
        "version": policy.concept_graph_version,
        "reviewVersion": policy.graph_review_version,
        "graphHash": policy.concept_graph_hash,
        "reviewHash": policy.graph_review_hash,
        "configHash": config.content_hash,
        "nodes": [{"id": c, "label": getattr(policy, "labels", LABELS).get(c, c)} for c in policy.nodes[topic]],
        "edges": [
            {"source": e.prerequisite, "target": e.dependent}
            for e in policy.accepted_edges
            if e.topic == topic
        ],
        "relation": "reviewed_prerequisite_candidate",
        "reviewerType": getattr(policy, "reviewer_type", "human"),
    }
