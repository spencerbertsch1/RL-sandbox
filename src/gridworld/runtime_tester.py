import pickle 
import plotly
import plotly.express as px 
import pandas as pd 
import numpy as np 

def main():

    with open('/Users/spencerbertsch/Desktop/dev/RL-sandbox/src/gridworld/metadata/burned_nodes.pkl', 'rb') as f:
        burned_nodes: list = pickle.load(f)    
    with open('/Users/spencerbertsch/Desktop/dev/RL-sandbox/src/gridworld/metadata/no_burned_nodes.pkl', 'rb') as f:
        no_burned_nodes: list = pickle.load(f)   

    df = pd.DataFrame({"Burned_Times": burned_nodes, "No_Burned_Times": no_burned_nodes})
    df['X'] = df.index

    # Plot 
    fig = px.line(df, x='X', y='Burned_Times')

    # Only thing I figured is - I could do this 
    fig.add_scatter(x=df['X'], y=df['No_Burned_Times'], mode='lines', showlegend=True)

    # Show plot 
    fig.show()

if __name__ == "__main__":
    main()
