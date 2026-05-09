#include "../event/event.h"
#include "../component.h"

// Component that listens for events
// Handles physics-related behavior and responds to collision events through the event system
class PhysicsComponent : public Component, public EventListener {
public:
    void Initialize() override {
        // Register as event listener
        GetEventSystem().AddListener(this);
    }

    ~PhysicsComponent() override {
        // Unregister as event listener
        GetEventSystem().RemoveListener(this);
    }

    void OnEvent(const Event& event) override {
        if (auto collisionEvent = dynamic_cast<const CollisionEvent*>(&event)) {
            // Handle collision event
        }
    }

private:
    EventSystem& GetEventSystem() {
        // Get event system from somewhere (e.g., service locator)
        static EventSystem eventSystem;
        return eventSystem;
    }
};
