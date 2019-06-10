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
                withCredentials([[$class: 'UsernamePasswordMultiBinding', credentialsId:'df9eaa47-de60-4e76-b6f8-8a490ce0bd49',
                                  usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD']]) {
                    sh 'git config --global user.name $USERNAME'
                    sh 'git config --global user.password $PASSWORD'
                    sh 'git submodule update --init --depth 1'
                }
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