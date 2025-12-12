load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "s4nnc",
    commit = "819daa5a9dcf7d10bfeb45de2b60a842b972ce83",
    remote = "https://github.com/liuliu/s4nnc.git",
    shallow_since = "1765519404 -0500",
)

load("@s4nnc//:deps.bzl", "s4nnc_deps")

s4nnc_deps()

load("@ccv//config:ccv.bzl", "ccv_deps", "ccv_setting")

ccv_deps()

load("@build_bazel_rules_cuda//gpus:cuda_configure.bzl", "cuda_configure")
load("@build_bazel_rules_cuda//nccl:nccl_configure.bzl", "nccl_configure")

cuda_configure(name = "local_config_cuda")

nccl_configure(name = "local_config_nccl")

ccv_setting(
    name = "local_config_ccv",
    have_cblas = True,
    have_cudnn = True,
    have_nccl = True,
    have_libjpeg = True,
    have_libpng = True,
    have_pthread = True,
    use_dispatch = True,
    use_openmp = True,
)

git_repository(
    name = "swift-fickling",
    commit = "f86bc6b694b3167e6e5be0cb6f16e5c5ba85be63",
    remote = "https://github.com/liuliu/swift-fickling.git",
    shallow_since = "1738984072 -0500",
)

load("@swift-fickling//:deps.bzl", "swift_fickling_deps")

swift_fickling_deps()

git_repository(
    name = "swift-sentencepiece",
    commit = "b0f9edd91fccbc4a1ba2323d82542beaec684fa5",
    remote = "https://github.com/liuliu/swift-sentencepiece.git",
    shallow_since = "1753399872 -0400",
)

load("@swift-sentencepiece//:deps.bzl", "swift_sentencepiece_deps")

swift_sentencepiece_deps()

new_git_repository(
    name = "SwiftPNG",
    build_file = "swift-png.BUILD",
    commit = "075dfb248ae327822635370e9d4f94a5d3fe93b2",
    remote = "https://github.com/kelvin13/swift-png",
    shallow_since = "1645648674 -0600",
)

new_git_repository(
    name = "ZIPFoundation",
    build_file = "zip-foundation.BUILD",
    commit = "642436f3684009ca7a5e3d6b30f2ecea26f8f772",
    remote = "https://github.com/weichsel/ZIPFoundation.git",
    shallow_since = "1665504317 +0200",
)

git_repository(
    name = "build_bazel_rules_swift",
    commit = "bffd22a56b8949616dfbd710cdca385cb2800274",
    remote = "https://github.com/bazelbuild/rules_swift.git",
    shallow_since = "1752542865 -0400",
)

load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")

swift_rules_dependencies()

load("@build_bazel_rules_swift//swift:extras.bzl", "swift_rules_extra_dependencies")

swift_rules_extra_dependencies()

http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.4.0/rules_python-0.4.0.tar.gz",
    sha256 = "954aa89b491be4a083304a2cb838019c8b8c3720a7abb9c4cb81ac7a24230cea",
)

load("@rules_python//python:pip.bzl", "pip_install")

new_git_repository(
    name = "SwiftArgumentParser",
    build_file = "swift-argument-parser.BUILD",
    commit = "9f39744e025c7d377987f30b03770805dcb0bcd1",
    remote = "https://github.com/apple/swift-argument-parser.git",
    shallow_since = "1661571047 -0500",
)

new_git_repository(
    name = "SwiftSystem",
    build_file = "swift-system.BUILD",
    commit = "025bcb1165deab2e20d4eaba79967ce73013f496",
    remote = "https://github.com/apple/swift-system.git",
    shallow_since = "1654977448 -0700",
)

new_git_repository(
    name = "SwiftToolsSupportCore",
    build_file = "swift-tools-support-core.BUILD",
    commit = "286b48b1d73388e1d49b2bb33aabf995838104e3",
    remote = "https://github.com/apple/swift-tools-support-core.git",
    shallow_since = "1670947584 -0800",
)

new_git_repository(
    name = "SwiftSyntax",
    build_file = "swift-syntax.BUILD",
    commit = "cd793adf5680e138bf2bcbaacc292490175d0dcd",
    remote = "https://github.com/apple/swift-syntax.git",
    shallow_since = "1676877517 +0100",
)

new_git_repository(
    name = "SwiftFormat",
    build_file = "swift-format.BUILD",
    commit = "9f1cc7172f100118229644619ce9c8f9ebc1032c",
    remote = "https://github.com/apple/swift-format.git",
    shallow_since = "1676404655 +0000",
)

load("@s4nnc//:deps.bzl", "s4nnc_extra_deps")

s4nnc_extra_deps()

# buildifier is written in Go and hence needs rules_go to be built.
# See https://github.com/bazelbuild/rules_go for the up to date setup instructions.
http_archive(
    name = "io_bazel_rules_go",
    sha256 = "6dc2da7ab4cf5d7bfc7c949776b1b7c733f05e56edc4bcd9022bb249d2e2a996",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.39.1/rules_go-v0.39.1.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.39.1/rules_go-v0.39.1.zip",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies")

go_rules_dependencies()

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains")

go_register_toolchains(version = "1.20.3")

http_archive(
    name = "bazel_gazelle",
    sha256 = "727f3e4edd96ea20c29e8c2ca9e8d2af724d8c7778e7923a854b2c80952bc405",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.30.0/bazel-gazelle-v0.30.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.30.0/bazel-gazelle-v0.30.0.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

# If you use WORKSPACE.bazel, use the following line instead of the bare gazelle_dependencies():
# gazelle_dependencies(go_repository_default_config = "@//:WORKSPACE.bazel")
gazelle_dependencies()

http_archive(
    name = "com_google_protobuf",
    sha256 = "3bd7828aa5af4b13b99c191e8b1e884ebfa9ad371b0ce264605d347f135d2568",
    strip_prefix = "protobuf-3.19.4",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v3.19.4.tar.gz",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

http_archive(
    name = "com_github_bazelbuild_buildtools",
    sha256 = "ae34c344514e08c23e90da0e2d6cb700fcd28e80c02e23e4d5715dddcb42f7b3",
    strip_prefix = "buildtools-4.2.2",
    urls = [
        "https://github.com/bazelbuild/buildtools/archive/refs/tags/4.2.2.tar.gz",
    ],
)
