# -*- coding: utf-8 -*-
"""
PrepStateManager模块单元测试

测试:
1. PrepStateKeys - 状态键常量
2. PrepStateManager - 状态管理器方法
3. 命名空间处理
4. 状态生命周期管理
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class MockSessionState(dict):
    """模拟Streamlit session_state"""
    pass


class TestPrepStateKeys:
    """PrepStateKeys常量测试"""

    def test_file_keys_exist(self):
        """测试文件相关键存在"""
        from dashboard.models.DFM.prep.ui.state import PrepStateKeys

        assert hasattr(PrepStateKeys, 'TRAINING_DATA_FILE')
        assert hasattr(PrepStateKeys, 'UPLOADED_FILE_PATH')
        assert hasattr(PrepStateKeys, 'FILE_BYTES')

    def test_param_keys_exist(self):
        """测试参数配置键存在"""
        from dashboard.models.DFM.prep.ui.state import PrepStateKeys

        assert hasattr(PrepStateKeys, 'PARAM_TARGET_FREQ')
        assert hasattr(PrepStateKeys, 'PARAM_DATA_START_DATE')
        assert hasattr(PrepStateKeys, 'PARAM_DATA_END_DATE')
        assert hasattr(PrepStateKeys, 'PARAM_ZERO_HANDLING')

    def test_result_keys_exist(self):
        """测试处理结果键存在"""
        from dashboard.models.DFM.prep.ui.state import PrepStateKeys

        assert hasattr(PrepStateKeys, 'PREPARED_DATA_DF')
        assert hasattr(PrepStateKeys, 'BASE_PREPARED_DATA_DF')
        assert hasattr(PrepStateKeys, 'TRANSFORM_LOG_OBJ')
        assert hasattr(PrepStateKeys, 'INDUSTRY_MAP_OBJ')

    def test_transform_keys_exist(self):
        """测试变量转换键存在"""
        from dashboard.models.DFM.prep.ui.state import PrepStateKeys

        assert hasattr(PrepStateKeys, 'TRANSFORM_CONFIG_DF')
        assert hasattr(PrepStateKeys, 'VARIABLE_TRANSFORM_DETAILS')
        assert hasattr(PrepStateKeys, 'VAR_NATURE_MAP_OBJ')

    def test_keys_are_strings(self):
        """测试所有键都是字符串"""
        from dashboard.models.DFM.prep.ui.state import PrepStateKeys

        keys = [
            PrepStateKeys.TRAINING_DATA_FILE,
            PrepStateKeys.PARAM_TARGET_FREQ,
            PrepStateKeys.PREPARED_DATA_DF,
        ]
        for key in keys:
            assert isinstance(key, str)

    def test_frozen_dataclass(self):
        """测试dataclass是frozen的"""
        from dashboard.models.DFM.prep.ui.state import PrepStateKeys

        # frozen=True意味着实例不可修改
        instance = PrepStateKeys()
        try:
            instance.TRAINING_DATA_FILE = "new_value"
            assert False, "应该抛出FrozenInstanceError"
        except Exception:
            pass  # 预期会抛出异常


class TestPrepStateManager:
    """PrepStateManager方法测试"""

    def setup_method(self):
        """每个测试方法前设置mock"""
        self.mock_session_state = MockSessionState()

    def test_full_key_generation(self):
        """测试完整键名生成"""
        from dashboard.models.DFM.prep.ui.state import PrepStateManager

        manager = PrepStateManager()

        # 测试_full_key方法
        assert manager._full_key("test_key") == "data_prep.test_key"
        # 已有前缀的不重复添加
        assert manager._full_key("data_prep.test_key") == "data_prep.test_key"

    def test_custom_namespace(self):
        """测试自定义命名空间"""
        from dashboard.models.DFM.prep.ui.state import PrepStateManager

        manager = PrepStateManager(namespace="custom_ns")
        assert manager._full_key("key") == "custom_ns.key"

    def test_get_cache_key(self):
        """测试缓存键生成"""
        from dashboard.models.DFM.prep.ui.state import PrepStateManager

        manager = PrepStateManager()
        cache_key = manager.get_cache_key("date_range", "test.xlsx", 1024)
        assert cache_key == "date_range_test.xlsx_1024"

    @patch('streamlit.session_state', new_callable=MockSessionState)
    def test_get_set_operations(self, mock_state):
        """测试get/set操作"""
        from dashboard.models.DFM.prep.ui.state import PrepStateManager

        manager = PrepStateManager()

        # 设置值
        manager.set("test_key", "test_value")
        assert mock_state["data_prep.test_key"] == "test_value"

        # 获取值
        value = manager.get("test_key")
        assert value == "test_value"

        # 获取不存在的键返回默认值
        default_value = manager.get("nonexistent", "default")
        assert default_value == "default"

    @patch('streamlit.session_state', new_callable=MockSessionState)
    def test_exists_operation(self, mock_state):
        """测试exists操作"""
        from dashboard.models.DFM.prep.ui.state import PrepStateManager

        manager = PrepStateManager()

        mock_state["data_prep.existing_key"] = "value"

        assert manager.exists("existing_key") is True
        assert manager.exists("nonexistent_key") is False

    @patch('streamlit.session_state', new_callable=MockSessionState)
    def test_delete_operation(self, mock_state):
        """测试delete操作"""
        from dashboard.models.DFM.prep.ui.state import PrepStateManager

        manager = PrepStateManager()

        mock_state["data_prep.to_delete"] = "value"
        assert "data_prep.to_delete" in mock_state

        result = manager.delete("to_delete")
        assert result is True
        assert "data_prep.to_delete" not in mock_state

        # 删除不存在的键返回False
        result = manager.delete("nonexistent")
        assert result is False


class TestStateLifecycle:
    """状态生命周期测试"""

    @patch('streamlit.session_state', new_callable=MockSessionState)
    def test_clear_results(self, mock_state):
        """测试清除结果状态"""
        from dashboard.models.DFM.prep.ui.state import PrepStateManager, PrepStateKeys

        manager = PrepStateManager()

        # 设置一些结果状态
        mock_state["data_prep." + PrepStateKeys.PREPARED_DATA_DF] = "data"
        mock_state["data_prep." + PrepStateKeys.TRANSFORM_LOG_OBJ] = "log"

        manager.clear_results()

        assert mock_state.get("data_prep." + PrepStateKeys.PREPARED_DATA_DF) is None
        assert mock_state.get("data_prep." + PrepStateKeys.TRANSFORM_LOG_OBJ) is None

    @patch('streamlit.session_state', new_callable=MockSessionState)
    def test_clear_params(self, mock_state):
        """测试清除参数状态"""
        from dashboard.models.DFM.prep.ui.state import PrepStateManager, PrepStateKeys

        manager = PrepStateManager()

        # 设置一些参数状态
        mock_state["data_prep." + PrepStateKeys.PARAM_TARGET_FREQ] = "W-FRI"
        mock_state["data_prep." + PrepStateKeys.PARAM_ZERO_HANDLING] = "missing"

        manager.clear_params()

        assert mock_state.get("data_prep." + PrepStateKeys.PARAM_TARGET_FREQ) is None
        assert mock_state.get("data_prep." + PrepStateKeys.PARAM_ZERO_HANDLING) is None

    @patch('streamlit.session_state', new_callable=MockSessionState)
    def test_on_file_change(self, mock_state):
        """测试文件变更处理"""
        from dashboard.models.DFM.prep.ui.state import PrepStateManager, PrepStateKeys

        manager = PrepStateManager()

        # 设置旧状态
        mock_state["data_prep." + PrepStateKeys.PARAM_TARGET_FREQ] = "old_value"
        mock_state["data_prep." + PrepStateKeys.PREPARED_DATA_DF] = "old_data"

        # 触发文件变更
        manager.on_file_change("new_file.xlsx", b"file_content")

        # 验证文件信息更新
        assert mock_state["data_prep." + PrepStateKeys.UPLOADED_FILE_PATH] == "new_file.xlsx"
        assert mock_state["data_prep." + PrepStateKeys.FILE_BYTES] == b"file_content"
        assert mock_state["data_prep." + PrepStateKeys.DATE_DETECTION_NEEDED] is True

        # 验证旧状态被清除
        assert mock_state.get("data_prep." + PrepStateKeys.PARAM_TARGET_FREQ) is None
        assert mock_state.get("data_prep." + PrepStateKeys.PREPARED_DATA_DF) is None


def run_tests():
    """运行所有测试"""
    import traceback

    test_classes = [TestPrepStateKeys, TestPrepStateManager, TestStateLifecycle]
    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        instance = test_class()
        print(f"\n{'='*50}")
        print(f"运行测试类: {test_class.__name__}")
        print('='*50)

        # 调用setup_method如果存在
        if hasattr(instance, 'setup_method'):
            instance.setup_method()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  [PASS] {method_name}")
                    passed_tests += 1
                except AssertionError as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))
                except Exception as e:
                    print(f"  [ERROR] {method_name}: {e}")
                    traceback.print_exc()
                    failed_tests.append((test_class.__name__, method_name, str(e)))

    print(f"\n{'='*50}")
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    print('='*50)

    if failed_tests:
        print("\n失败的测试:")
        for class_name, method, error in failed_tests:
            print(f"  - {class_name}.{method}: {error}")
        return 1
    return 0


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
