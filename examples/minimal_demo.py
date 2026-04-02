from sb import SBNetwork, Signal, SignalTemplate, Space, TransformRule


def build_demo_network() -> SBNetwork:
    spaces = [
        Space(
            space_id="cat_space",
            space_type="concept",
            sensitive_tags=frozenset({"animal", "cat"}),
            preferred_kinds=frozenset({"entity"}),
            transform_rules=[
                TransformRule(
                    required_tags=frozenset({"cat"}),
                    emitted=(
                        SignalTemplate(
                            kind="attribute",
                            value="agile",
                            tags=("animal", "motion_hint"),
                            confidence_gain=0.10,
                        ),
                    ),
                )
            ],
        ),
        Space(
            space_id="chase_space",
            space_type="operator",
            sensitive_tags=frozenset({"action", "chase", "motion_hint"}),
            preferred_kinds=frozenset({"action", "attribute"}),
            transform_rules=[
                TransformRule(
                    required_tags=frozenset({"chase", "motion_hint"}),
                    emitted=(
                        SignalTemplate(
                            kind="relation",
                            value="pursuit",
                            tags=("dynamic_scene", "targeting", "action", "chase", "pursuit"),
                            confidence_gain=0.15,
                        ),
                    ),
                )
            ],
        ),
        Space(
            space_id="play_scene",
            space_type="episode",
            sensitive_tags=frozenset({"ball", "pursuit", "dynamic_scene"}),
            preferred_kinds=frozenset({"entity", "relation"}),
            activation_bias=0.15,
            transform_rules=[
                TransformRule(
                    required_tags=frozenset({"ball", "pursuit"}),
                    emitted=(
                        SignalTemplate(
                            kind="scene",
                            value="cat_playing_with_ball",
                            tags=("play", "coherent_answer"),
                            confidence_gain=0.30,
                        ),
                    ),
                )
            ],
        ),
    ]
    network = SBNetwork(spaces)
    network.reinforce_transition("cat_space", "chase_space", reward=0.20)
    network.reinforce_transition("chase_space", "play_scene", reward=0.25)
    return network


def main() -> None:
    network = build_demo_network()
    inputs = [
        Signal("s1", "entity", "cat", tags=("animal", "cat"), confidence=0.9, ttl=3),
        Signal("s2", "action", "chase", tags=("action", "chase"), confidence=0.8, ttl=3),
        Signal("s3", "entity", "ball", tags=("object", "ball"), confidence=0.85, ttl=3),
    ]
    result = network.infer(
        inputs,
        steps=4,
        beam_width=8,
        top_k=3,
        correlation_threshold=0.15,
        max_expansions=64,
    )

    print("Best score:", round(result.score, 3))
    print("Trace:")
    for item in result.trace:
        print(" -", item)

    print("Signals:")
    for signal in sorted(result.signals, key=lambda current: current.confidence, reverse=True):
        print(
            f" - {signal.kind:9s} value={signal.value!r} "
            f"conf={signal.confidence:.2f} ttl={signal.ttl} tags={signal.tags}"
        )


if __name__ == "__main__":
    main()
