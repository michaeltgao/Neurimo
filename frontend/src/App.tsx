import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import SignInPage from "./pages/SignInPage";
import ChildrenListPage from "./pages/ChildrenListPage";
import ChildDetailPage from "./pages/ChildDetailPage";
import VisitPage from "./pages/VisitPage";
import QuestionnairePage from "./pages/QuestionnairePage";
import ReportPage from "./pages/ReportPage";
import AssistedReviewPage from "./pages/AssistedReviewPage";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) {
    return <Navigate to="/signin" replace />;
  }
  return <>{children}</>;
}

function AppRoutes() {
  const { isAuthenticated } = useAuth();

  return (
    <Routes>
      <Route
        path="/signin"
        element={isAuthenticated ? <Navigate to="/children" replace /> : <SignInPage />}
      />
      <Route path="/" element={<Navigate to="/children" replace />} />
      <Route
        path="/children"
        element={
          <ProtectedRoute>
            <ChildrenListPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/children/:childId"
        element={
          <ProtectedRoute>
            <ChildDetailPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/visits/:visitId"
        element={
          <ProtectedRoute>
            <VisitPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/visits/:visitId/questionnaire"
        element={
          <ProtectedRoute>
            <QuestionnairePage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/visits/:visitId/report"
        element={
          <ProtectedRoute>
            <ReportPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/videos/:videoId/assisted-review"
        element={
          <ProtectedRoute>
            <AssistedReviewPage />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
