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
                source activate ~/miniconda/envs/odometry
                python -m unittest discover -s tests
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}