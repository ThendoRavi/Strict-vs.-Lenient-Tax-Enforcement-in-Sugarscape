import pynetlogo
import time

def test_connection():
    """Simple test to check if Python can connect to NetLogo"""
    
    print("🔄 Starting connection test...")
    
    try:
        # Step 1: Connect to NetLogo
        print("📡 Connecting to NetLogo...")
        netlogo = pynetlogo.NetLogoLink(gui=True)  # gui=True so you can see what's happening
        print("✅ Connected successfully!")
        
        # Step 2: Load the model
        print("📂 Loading NetLogo model...")
        netlogo.load_model('Sugarscape 2 Constant Growback.nlogo')
        print("✅ Model loaded successfully!")
        
        # Step 3: Setup the simulation
        print("🎯 Running setup...")
        netlogo.command('setup')
        print("✅ Setup complete!")
        
        # Step 4: Check if we have turtles
        population = netlogo.report('get-population')
        population_int = int(float(population))  # Convert JDouble to Python int via float
        print(f"👥 Population: {population_int} turtles")
        
        # Step 5: Get some basic info
        print("📊 Getting turtle states...")
        states = netlogo.report('report-states')
        rewards = netlogo.report('report-rewards')
        
        print(f"📝 Got {len(states)} turtle states")
        print(f"💰 Got {len(rewards)} turtle rewards")
        
        # Step 6: Show first few turtles
        print("\n🐢 First 3 turtles:")
        for i in range(min(3, len(states))):
            turtle_id, sugar_level, punished, history = states[i]
            reward = rewards[i]
            print(f"  Turtle {turtle_id}: Sugar Level={sugar_level}, Reward={reward:.1f}, Punished={punished}")
        
        # Step 7: Test sending simple actions
        print("\n🎮 Testing action sending...")
        # Send action 0 (pay) to all turtles
        simple_actions = [2] * population_int  # All turtles will "pay"
        action_string = "[" + " ".join(map(str, simple_actions)) + "]"
        
        print(f"� Sending actions: {action_string[:50]}...")
        netlogo.command(f'receive-actions {action_string}')
        print("✅ Actions sent successfully!")
        
        # Step 8: Run simulation for 100 ticks
        print("⏭️ Running simulation for 100 ticks...")
        num_ticks = 100
        
        for tick in range(num_ticks):
            # Send actions for this tick (all turtles will "pay")
            current_population = netlogo.report('get-population')
            current_population_int = int(float(current_population))
            
            if current_population_int == 0:
                print(f"❌ Population died out at tick {tick}")
                break
            
            # Send actions to all current turtles
            tick_actions = [2] * current_population_int  # All turtles will "pay"
            action_string = "[" + " ".join(map(str, tick_actions)) + "]"
            netlogo.command(f'receive-actions {action_string}')
            
            # Run one step
            netlogo.command('go')
            
            # Print progress every 20 ticks
            if (tick + 1) % 20 == 0:
                temp_population = netlogo.report('get-population')
                temp_population_int = int(float(temp_population))
                if temp_population_int > 0:
                    temp_rewards = netlogo.report('report-rewards')
                    avg_sugar = sum(temp_rewards) / len(temp_rewards)
                    print(f"  Tick {tick + 1:3d}: Population={temp_population_int}, Avg Sugar={avg_sugar:.1f}")
        
        print("✅ Simulation complete!")
        
        # Step 9: Check final results
        final_population = netlogo.report('get-population')
        final_population_int = int(float(final_population))  # Convert JDouble to Python int via float
        print(f"👥 Final population: {final_population_int} turtles")
        
        if final_population_int > 0:
            final_rewards = netlogo.report('report-rewards')
            print(f"💰 Final average sugar: {sum(final_rewards)/len(final_rewards):.1f}")
        else:
            print("💀 All turtles died during simulation")
        
        print("\n🎉 CONNECTION TEST SUCCESSFUL! 🎉")
        print("✅ Python can talk to NetLogo")
        print("✅ NetLogo can receive actions")
        print("✅ NetLogo can send back data")
        
        # Step 10: Clean up
        print("\n🧹 Cleaning up...")
        netlogo.kill_workspace()
        print("✅ Connection closed")
        
    except FileNotFoundError:
        print("❌ ERROR: NetLogo model file 'Sugarscape 2 Constant Growback.nlogo' not found!")
        print("   Make sure the .nlogo file is in the same folder as this Python script")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Make sure NetLogo is installed")
        print("2. Make sure pynetlogo is installed: pip install pynetlogo")
        print("3. Make sure the .nlogo file is in the same folder")
        print("4. Try closing NetLogo if it's already open")

if __name__ == "__main__":
    test_connection()