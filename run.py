import numpy as np

from misc import *

def main():
   
    UserInputsDict = user_inputs() #get user inputs
    
    LogStr = UserInputsDict.get('LogStr')
    ResumeTraining = UserInputsDict.get('ResumeTraining')    
    TrainingSteps = UserInputsDict.get('TrainingSteps')    
    SelModeInd = UserInputsDict.get('SelModeInd')    

    if SelModeInd == 1: #plot trained agent results

        plot_train(UserInputsDict)

    else:

        if not LogStr: #there are no log dirs inside algorithm-environment folder

            create_path(UserInputsDict) #create directories

        Env = create_env(UserInputsDict) #create gym environment
        MaxStepsPerEpisode = Env.env.spec.max_episode_steps

        if SelModeInd == 2: #train agent

            if ResumeTraining: #resume training

                Model = load_model(UserInputsDict, Env) #load existing model

            else: #new training

                Model = create_model(UserInputsDict, Env) #create new model
        
            Callback = create_callback(UserInputsDict, Env)
            print("\nThe training has just started!\n")
            Model.learn(total_timesteps=TrainingSteps, callback=Callback, \
                        log_interval=int(max(min(0.05 * TrainingSteps, \
                                     10 * MaxStepsPerEpisode), MaxStepsPerEpisode)))
            print("\nThe training has just finished!\n")
            save_model(UserInputsDict, Model, Env)
            rename_files(UserInputsDict)
            plot_train(UserInputsDict)

        elif SelModeInd == 0: #deploy trained agent

            Model = load_model(UserInputsDict, Env)
            print("\nThe deployment has just started!\n")
            Times, States, Actions = deploy_model(UserInputsDict, Model, Env)
            print("The deployment has just finished!\n")
            plot_deploy(UserInputsDict, Times, States, Actions)

if __name__ == "__main__":

    main()
