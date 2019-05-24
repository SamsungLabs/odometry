pipeline {
    agent {
        docker {image "odometry"}
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
                sh '~/miniconda/bin/activate ~/miniconda/envs/odometry'
                sh '~/miniconda/envs/odometry/bin/python -m unittest discover -s tests'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}