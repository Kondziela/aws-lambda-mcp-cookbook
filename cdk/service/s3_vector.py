from aws_cdk import CfnOutput, Duration, RemovalPolicy
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as _lambda
from aws_cdk.aws_lambda_python_alpha import PythonLayerVersion
from aws_cdk.aws_logs import RetentionDays
from constructs import Construct

import cdk.service.constants as constants


class S3VectorConstruct(Construct):
    def __init__(self, scope: Construct, id_: str) -> None:
        super().__init__(scope, id_)
        self.id_ = id_
        self.lambda_role = self._build_lambda_role()
        self._grant_permissions_to_lambda_role()
        self.common_layer = self._build_common_layer()
        self.lambda_function = self._build_lambda_function()
        self.query_lambda_function = self._build_query_lambda_function()

    def _build_lambda_role(self) -> iam.Role:
        return iam.Role(
            self,
            f'{self.id_}LambdaRole',
            assumed_by=iam.ServicePrincipal('lambda.amazonaws.com'),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(managed_policy_name=f'service-role/{constants.LAMBDA_BASIC_EXECUTION_ROLE}')
            ],
        )

    def _grant_permissions_to_lambda_role(self) -> None:
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    'bedrock:InvokeModel',
                ],
                resources=['arn:aws:bedrock:eu-central-1::foundation-model/amazon.titan-embed-text-v2:0'],
            )
        )
        self.lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    's3vectors:PutVectors',
                    's3vectors:QueryVectors',
                    's3vectors:GetVectors',
                ],
                resources=["*"]
            )
        )

    def _build_lambda_function(self) -> _lambda.Function:
        lambda_function = _lambda.Function(
            self,
            f'{self.id_}Lambda',
            runtime=_lambda.Runtime.PYTHON_3_13,
            code=_lambda.Code.from_asset(constants.BUILD_FOLDER),
            handler='service.s3_vector.s3_vector.lambda_handler',
            environment={
                constants.POWERTOOLS_SERVICE_NAME: f'{self.id_}Service',
                constants.POWER_TOOLS_LOG_LEVEL: 'INFO',
                'S3_VECTOR_REGION': 'eu-central-1',
                'BEDROCK_MODEL_ID': 'amazon.titan-embed-text-v2:0',
                'BEDROCK_REGION': 'eu-central-1',
                'S3_VECTOR_BUCKET': 'my-test-vector',  # Configure your S3 Vector bucket name
                'S3_VECTOR_INDEX_NAME': 'my-test-vector-index',  # Configure your S3 Vector index name
            },
            tracing=_lambda.Tracing.ACTIVE,
            retry_attempts=0,
            timeout=Duration.seconds(constants.API_HANDLER_LAMBDA_TIMEOUT),
            memory_size=constants.API_HANDLER_LAMBDA_MEMORY_SIZE,
            layers=[self.common_layer],
            role=self.lambda_role,
            log_retention=RetentionDays.ONE_DAY,
            logging_format=_lambda.LoggingFormat.JSON,
            system_log_level_v2=_lambda.SystemLogLevel.WARN,
            architecture=_lambda.Architecture.X86_64,
        )

        CfnOutput(self, f'{self.id_}LambdaOutput', value=lambda_function.function_name).override_logical_id(f'{self.id_}LambdaOutput')

        return lambda_function

    def _build_query_lambda_function(self) -> _lambda.Function:
        query_lambda_function = _lambda.Function(
            self,
            f'{self.id_}QueryLambda',
            runtime=_lambda.Runtime.PYTHON_3_13,
            code=_lambda.Code.from_asset(constants.BUILD_FOLDER),
            handler='service.s3_vector.s3_vector_query.lambda_handler',
            environment={
                constants.POWERTOOLS_SERVICE_NAME: f'{self.id_}QueryService',
                constants.POWER_TOOLS_LOG_LEVEL: 'INFO',
                'S3_VECTOR_REGION': 'eu-central-1',
                'BEDROCK_MODEL_ID': 'amazon.titan-embed-text-v2:0',
                'BEDROCK_REGION': 'eu-central-1',
                'S3_VECTOR_BUCKET': 'your-vector-bucket-name',  # Configure your S3 Vector bucket name
                'S3_VECTOR_INDEX_NAME': 'default-index',  # Configure your S3 Vector index name
            },
            tracing=_lambda.Tracing.ACTIVE,
            retry_attempts=0,
            timeout=Duration.seconds(constants.API_HANDLER_LAMBDA_TIMEOUT),
            memory_size=constants.API_HANDLER_LAMBDA_MEMORY_SIZE,
            layers=[self.common_layer],
            role=self.lambda_role,
            log_retention=RetentionDays.ONE_DAY,
            logging_format=_lambda.LoggingFormat.JSON,
            system_log_level_v2=_lambda.SystemLogLevel.WARN,
            architecture=_lambda.Architecture.X86_64,
        )

        CfnOutput(
            self,
            f'{self.id_}QueryLambdaOutput',
            value=query_lambda_function.function_name
        ).override_logical_id(f'{self.id_}QueryLambdaOutput')

        return query_lambda_function

    def _build_common_layer(self) -> PythonLayerVersion:
        return PythonLayerVersion(
            self,
            f'{self.id_}{constants.LAMBDA_LAYER_NAME}',
            entry=constants.COMMON_LAYER_BUILD_FOLDER,
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_13],
            removal_policy=RemovalPolicy.DESTROY,
            compatible_architectures=[_lambda.Architecture.X86_64],
        )
