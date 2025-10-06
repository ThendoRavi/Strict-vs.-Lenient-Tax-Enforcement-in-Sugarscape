"""
Test script to verify that agent positions are being correctly reported from NetLogo
"""
import pynetlogo
import numpy as np

def test_position_reporting():
    """Test the NetLogo position reporting functionality"""
    print("🔗 Testing NetLogo position reporting...")
    
    try:
        # Initialize NetLogo
        netlogo = pynetlogo.NetLogoLink(gui=False)
        netlogo.load_model('Sugarscape 2 Constant Growback.nlogo')
        
        print("✅ NetLogo model loaded successfully")
        
        # Setup simulation
        netlogo.command('setup')
        print("✅ Setup complete")
        
        # Get states
        states = netlogo.report('report-states')
        print(f"📊 Retrieved {len(states)} agent states")
        
        # Check the first few agents
        print("\n🐢 First 5 agents with positions:")
        for i, state in enumerate(states[:5]):
            if len(state) >= 10:
                agent_id = state[0]
                sugar_level = state[1]
                x_pos = state[8]
                y_pos = state[9]
                print(f"  Agent {agent_id}: Sugar={sugar_level}, Position=({x_pos}, {y_pos})")
            else:
                print(f"  Agent {state[0]}: State data incomplete ({len(state)} elements)")
        
        # Verify positions are within expected bounds (0-49 for a 50x50 grid)
        valid_positions = 0
        total_agents = len(states)
        
        for state in states:
            if len(state) >= 10:
                x_pos, y_pos = state[8], state[9]
                if 0 <= x_pos <= 49 and 0 <= y_pos <= 49:
                    valid_positions += 1
        
        print(f"\n📍 Position validation:")
        print(f"   Total agents: {total_agents}")
        print(f"   Valid positions: {valid_positions}")
        print(f"   Validation rate: {valid_positions/total_agents*100:.1f}%")
        
        # Test uniqueness of positions (agents shouldn't stack exactly)
        positions = [(state[8], state[9]) for state in states if len(state) >= 10]
        unique_positions = len(set(positions))
        print(f"   Unique positions: {unique_positions}")
        print(f"   Position diversity: {unique_positions/len(positions)*100:.1f}%")
        
        if valid_positions == total_agents:
            print("✅ All agent positions are within valid bounds!")
        else:
            print(f"⚠️  {total_agents - valid_positions} agents have invalid positions")
            
        netlogo.kill_workspace()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during position reporting test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_position_reporting()