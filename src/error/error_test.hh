#pragma once

namespace errors
{

    class error_test_t{
    private:
        bool m_error;
    public:
        error_test_t(bool error = true) : m_error(error) {}

        template<class T>
        bool operator()(T&& value) const
        {
            return m_error == (bool)std::forward<T>(value).error();
        }

        error_test_t operator==(bool test) const
        {
            return error_test_t(test ? m_error : !m_error);
        }

        error_test_t operator!() const
        {
            return error_test_t(!m_error);
        }
    };

    error_test_t error(true);
    error_test_t not_error(false);

} // namespace error