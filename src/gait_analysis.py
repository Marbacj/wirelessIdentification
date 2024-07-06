import numpy as np

def analyze_gait(steps, features):
    """分析步态特征"""
    gait_profiles = []
    for step, feature in zip(steps, features):
        gait_profile = compute_gait_profile(step, feature)
        gait_profiles.append(gait_profile)
    return gait_profiles

def compute_gait_profile(step, feature):
    # 示例：计算步态特征
    return np.concatenate([step, feature])

if __name__ == "__main__":
    steps = np.load('steps.npy')
    features = np.load('features.npy')
    gait_profiles = analyze_gait(steps, features)
    np.save('gait_profiles.npy', gait_profiles)
