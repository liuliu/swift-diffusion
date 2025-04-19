load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "s4nnc",
    commit = "fc2e4f0e7add4d9ebf00becc76731bdab298cac1",
    remote = "https://github.com/liuliu/s4nnc.git",
    shallow_since = "1745021596 -0400",
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
    commit = "296c8eb774332a3a49c8c403fdbec373d9fb2f96",
    remote = "https://github.com/liuliu/swift-fickling.git",
    shallow_since = "1675031846 -0500",
)

load("@swift-fickling//:deps.bzl", "swift_fickling_deps")

swift_fickling_deps()

git_repository(
    name = "swift-sentencepiece",
    commit = "2c4ec57bea836f8b420179ee7670304a4972c572",
    remote = "https://github.com/liuliu/swift-sentencepiece.git",
    shallow_since = "1683864360 -0400",
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
    commit = "3bc7bc164020a842ae08e0cf071ed35f0939dd39",
    remote = "https://github.com/bazelbuild/rules_swift.git",
    shallow_since = "1654173801 -0500",
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
    sha256 = "099a9fb96a376ccbbb7d291ed4ecbdfd42f6bc822ab77ae6f1b5cb9e914e94fa",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.19.1")

http_archive(
    name = "bazel_gazelle",
    sha256 = "501deb3d5695ab658e82f6f6f549ba681ea3ca2a5fb7911154b5aa45596183fa",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.26.0/bazel-gazelle-v0.26.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.26.0/bazel-gazelle-v0.26.0.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

git_repository(
    name = "com_github_bazelbuild_buildtools",
    commit = "174cbb4ba7d15a3ad029c2e4ee4f30ea4d76edce",
    remote = "https://github.com/bazelbuild/buildtools.git",
    shallow_since = "1607975103 +0100",
)
