pipeline {
    agent {
        nvidia-docker {
        image 'odometry'
        args '-v /datasets:/datasets'
        }
    }
    stages {
        stage('Build') {
            steps {
                echo 'Building..'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
                sh 'nvidia-smi'
                sh '~/miniconda/bin/activate ~/miniconda/envs/odometry'
                sh '~/miniconda/envs/odometry/bin/python -m unittest discover -s tests'
            }
        }
    }
}