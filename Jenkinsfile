pipeline {
    agent {
        dockerfile {
        args '--runtime=nvidia -v /datasets:/datasets'
        }
    }
    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                sh 'cd odometry'
                withCredentials([[$class: 'UsernamePasswordMultiBinding', credentialsId:'df9eaa47-de60-4e76-b6f8-8a490ce0bd49',
                                  usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD']]) {
                    sh 'git submodule update --init --recursive -u $USERNAME -p $PASSWORD'
                }
                sh 'git submodule update --depth 1 --init --remote submodules/tf_models'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
                sh 'nvidia-smi'
                sh '/home/odometry/miniconda/bin/activate /home/odometry/miniconda/envs/odometry'
                sh '/home/odometry/miniconda/envs/odometry/bin/python -m unittest discover -s tests'
            }
        }
    }
}