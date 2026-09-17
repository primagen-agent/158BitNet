#include "memory_snapshot.h"
#include "sha256.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static int path_join(char *out, size_t size, const char *base, const char *suffix) {
    int count = snprintf(out, size, "%s%s", base, suffix);
    return count >= 0 && (size_t)count < size ? 0 : -1;
}

static int digest_hex(const char *path, char hex[65]) {
    unsigned char digest[32];
    static const char digits[] = "0123456789abcdef";
    if (bitnet_sha256_file(path, digest) != 0) return -1;
    for (size_t i = 0; i < 32; ++i) {
        hex[i * 2] = digits[digest[i] >> 4];
        hex[i * 2 + 1] = digits[digest[i] & 15];
    }
    hex[64] = '\0';
    return 0;
}

static int sync_file(const char *path) {
    FILE *file = fopen(path, "rb+");
    int status;
    if (file == NULL) return -1;
#ifdef _WIN32
    status = _commit(_fileno(file));
#else
    status = fsync(fileno(file));
#endif
    if (fclose(file) != 0) status = -1;
    return status;
}

static int sync_parent(const char *path) {
#ifdef _WIN32
    (void)path;
    return 0;
#else
    char directory[1200];
    char *slash;
    int fd, status;
    if (path_join(directory, sizeof directory, path, "") != 0) return -1;
    slash = strrchr(directory, '/');
    if (slash == directory) slash[1] = '\0';
    else if (slash != NULL) *slash = '\0';
    else strcpy(directory, ".");
    fd = open(directory, O_RDONLY);
    if (fd < 0) return -1;
    status = fsync(fd);
    close(fd);
    return status;
#endif
}

static int replace_file(const char *source, const char *target) {
#ifdef _WIN32
    return MoveFileExA(source, target,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(source, target);
#endif
}

static int content_path(char *out, size_t size, const char *base,
                        const char *hash, const char *extension) {
    int n = snprintf(out, size, "%s.%s%s", base, hash, extension);
    return n >= 0 && (size_t)n < size ? 0 : -1;
}

int metis_memory_snapshot_paths(const char *base,
    char *episodes, size_t episodes_size, char *events, size_t events_size) {
    return metis_memory_snapshot_paths_resident(base, episodes, episodes_size,
        events, events_size, NULL, 0);
}

int metis_memory_snapshot_paths_resident(const char *base,
    char *episodes, size_t episodes_size, char *events, size_t events_size,
    char *resident, size_t resident_size) {
    char manifest[1200], body[206], hashes[3][65], actual[65];
    FILE *file;
    if (base == NULL || episodes == NULL || events == NULL ||
        path_join(manifest, sizeof manifest, base, ".bnsnapshot") != 0) return -1;
    file = fopen(manifest, "rb");
    if (file == NULL) {
        if (errno != ENOENT || resident != NULL) return -1;
        return path_join(episodes, episodes_size, base, ".bnepisodic") ||
               path_join(events, events_size, base, ".bnevent") ? -1 : 0;
    }
    /* 9-byte magic line, two 64-byte SHA-256 lines. */
    size_t count = fread(body, 1, sizeof body, file);
    int failed = ferror(file);
    fclose(file);
    int parts = resident == NULL ? 2 : 3;
    if (failed || count != 9u + (size_t)parts * 65u ||
        memcmp(body, resident == NULL ? "BNMSNAP1\n" : "BNMSNAP2\n", 9) != 0) return -1;
    for (int h = 0; h < parts; ++h) {
        if (body[9 + h * 65 + 64] != '\n') return -1;
        memcpy(hashes[h], body + 9 + h * 65, 64);
        hashes[h][64] = '\0';
        if (strspn(hashes[h], "0123456789abcdef") != 64) return -1;
    }
    if (content_path(episodes, episodes_size, base, hashes[0], ".bnepisodic") ||
        content_path(events, events_size, base, hashes[1], ".bnevent") ||
        digest_hex(episodes, actual) || strcmp(actual, hashes[0]) ||
        digest_hex(events, actual) || strcmp(actual, hashes[1])) return -1;
    if (resident != NULL &&
        (content_path(resident, resident_size, base, hashes[2], ".bnresident") ||
         digest_hex(resident, actual) || strcmp(actual, hashes[2]))) return -1;
    return 0;
}

int metis_memory_snapshot_save(const char *base,
    const metis_episodic_store_t *episodes, const metis_event_store_t *events) {
    return metis_memory_snapshot_save_resident(base, episodes, events, NULL, NULL);
}

int metis_memory_snapshot_save_resident(const char *base,
    const metis_episodic_store_t *episodes, const metis_event_store_t *events,
    const resident_state_t *resident, const resident_model_t *model) {
    char ep_temp[1200], ev_temp[1200], manifest_temp[1200], manifest[1200];
    char ep_path[1200], ev_path[1200], ep_hash[65], ev_hash[65];
    char rs_temp[1200], rs_path[1200], rs_hash[65];
    FILE *file = NULL;
    int status = -1;
    if (base == NULL || episodes == NULL || events == NULL ||
        ((resident == NULL) != (model == NULL)) ||
        (resident != NULL && resident->count != events->count) ||
        path_join(rs_temp, sizeof rs_temp, base, ".pending.bnresident") ||
        path_join(ep_temp, sizeof ep_temp, base, ".pending.bnepisodic") ||
        path_join(ev_temp, sizeof ev_temp, base, ".pending.bnevent") ||
        path_join(manifest_temp, sizeof manifest_temp, base, ".pending.bnsnapshot") ||
        path_join(manifest, sizeof manifest, base, ".bnsnapshot")) return -1;
    if (metis_episodic_save(episodes, ep_temp) ||
        metis_event_store_save(events, ev_temp) ||
        sync_file(ep_temp) || sync_file(ev_temp) ||
        digest_hex(ep_temp, ep_hash) || digest_hex(ev_temp, ev_hash) ||
        content_path(ep_path, sizeof ep_path, base, ep_hash, ".bnepisodic") ||
        content_path(ev_path, sizeof ev_path, base, ev_hash, ".bnevent") ||
        replace_file(ep_temp, ep_path) || replace_file(ev_temp, ev_path) ||
        sync_parent(base)) goto cleanup;
    if (resident != NULL &&
        (resident_state_save(resident, model, rs_temp) || sync_file(rs_temp) ||
         digest_hex(rs_temp, rs_hash) ||
         content_path(rs_path, sizeof rs_path, base, rs_hash, ".bnresident") ||
         replace_file(rs_temp, rs_path) || sync_parent(base))) goto cleanup;
    file = fopen(manifest_temp, "wb");
    if (file == NULL) goto cleanup;
    int written = resident == NULL ?
        fprintf(file, "BNMSNAP1\n%s\n%s\n", ep_hash, ev_hash) :
        fprintf(file, "BNMSNAP2\n%s\n%s\n%s\n", ep_hash, ev_hash, rs_hash);
    int closed = fclose(file);
    file = NULL;
    if (written != (resident == NULL ? 139 : 204) || closed != 0 || sync_file(manifest_temp) ||
        replace_file(manifest_temp, manifest)) goto cleanup;
    /* A failure here leaves either the old or the complete new generation. */
    status = sync_parent(base);
cleanup:
    if (file != NULL) fclose(file);
    remove(ep_temp);
    remove(ev_temp);
    remove(manifest_temp);
    if (resident != NULL) remove(rs_temp);
    return status;
}
