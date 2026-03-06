
import argparse
import json
import random
import re

YELP_PATH = 'yelp/yelp_academic_dataset_business.json'
OUTPUT_PATH = 'data/synthetic_golden_dataset_2k.json'


def clamp(value, low, high):
    return max(low, min(high, value))


def weighted_choice(options):
    """Pick one key from [(key, weight), ...]."""
    total = sum(weight for _, weight in options)
    draw = random.random() * total
    acc = 0.0
    for key, weight in options:
        acc += weight
        if draw <= acc:
            return key
    return options[-1][0]


def random_phone_us():
    return f"+1{random.randint(200,999)}{random.randint(200,999)}{random.randint(1000,9999)}"


def random_website(name):
    clean = re.sub(r'[^a-zA-Z0-9]', '', name).lower() or 'business'
    return f"https://www.{clean}.com"


def normalize_categories(categories):
    if not categories:
        return []
    if isinstance(categories, str):
        parts = [p.strip() for p in categories.split(',') if p.strip()]
        return parts
    return []


def perturb_name(name, severity):
    value = name or ''
    if not value:
        return value
    if severity < 0.2:
        return value

    ops = [
        ('lower', 0.25),
        ('upper', 0.20),
        ('drop_punct', 0.20),
        ('drop_suffix', 0.20),
        ('truncate_branch', 0.15),
    ]
    op = weighted_choice(ops)

    if op == 'lower':
        return value.lower()
    if op == 'upper':
        return value.upper()
    if op == 'drop_punct':
        return re.sub(r"[\'\.,]", '', value)
    if op == 'drop_suffix':
        return re.sub(r'\b(inc|llc|ltd|corp|co)\.?\b', '', value, flags=re.IGNORECASE).strip()
    if op == 'truncate_branch':
        tokens = value.split()
        return ' '.join(tokens[:-1]).strip() if len(tokens) > 2 else value
    return value


def perturb_phone(phone, severity):
    value = phone or ''
    if not value:
        return ''
    digits = ''.join(ch for ch in value if ch.isdigit())
    if not digits:
        return value
    if severity < 0.2:
        return value

    op = weighted_choice([
        ('digits_only', 0.35),
        ('us_format', 0.30),
        ('drop_country', 0.20),
        ('digit_typo', 0.15),
    ])

    if op == 'digits_only':
        return digits
    if op == 'us_format' and len(digits) >= 10:
        last = digits[-10:]
        return f"({last[0:3]}) {last[3:6]}-{last[6:10]}"
    if op == 'drop_country' and len(digits) > 10:
        return digits[-10:]
    if op == 'digit_typo' and len(digits) >= 7:
        idx = random.randint(0, len(digits) - 1)
        replacement = str((int(digits[idx]) + random.randint(1, 8)) % 10)
        return digits[:idx] + replacement + digits[idx + 1:]
    return value


def perturb_website(url, severity):
    value = url or ''
    if not value:
        return ''
    if severity < 0.2:
        return value

    op = weighted_choice([
        ('http_only', 0.35),
        ('drop_www', 0.25),
        ('add_tracking', 0.20),
        ('drop_site', 0.20),
    ])

    if op == 'http_only':
        return value.replace('https://', 'http://')
    if op == 'drop_www':
        return value.replace('://www.', '://')
    if op == 'add_tracking':
        return value.rstrip('/') + '/?utm_source=directory'
    if op == 'drop_site':
        return ''
    return value


def perturb_address(addr_obj, severity):
    if severity < 0.2:
        return dict(addr_obj)

    out = dict(addr_obj)
    op = weighted_choice([
        ('drop_postcode', 0.35),
        ('drop_region', 0.25),
        ('abbrev_freeform', 0.25),
        ('drop_unit_like', 0.15),
    ])

    if op == 'drop_postcode':
        out.pop('postcode', None)
    elif op == 'drop_region':
        out.pop('region', None)
    elif op == 'abbrev_freeform':
        freeform = out.get('freeform', '')
        freeform = freeform.replace(' Street', ' St').replace(' Avenue', ' Ave')
        freeform = freeform.replace(' Road', ' Rd').replace(' Boulevard', ' Blvd')
        out['freeform'] = freeform
    elif op == 'drop_unit_like':
        freeform = out.get('freeform', '')
        out['freeform'] = re.sub(r'\b(unit|suite|ste)\s*\w+\b', '', freeform, flags=re.IGNORECASE).strip(', ')

    return out


def perturb_category(category_primary, alternates, severity):
    primary = category_primary or ''
    if severity < 0.2:
        return primary, alternates

    op = weighted_choice([
        ('flatten', 0.35),
        ('lowercase', 0.30),
        ('drop_alternates', 0.20),
        ('generic', 0.15),
    ])

    if op == 'flatten':
        return primary.split(' > ')[0], []
    if op == 'lowercase':
        return primary.lower(), alternates
    if op == 'drop_alternates':
        return primary, []
    if op == 'generic':
        generic = primary.split(' > ')[0] if primary else 'business'
        return generic, []
    return primary, alternates


def make_side(clean, severity):
    """Create one side of a pair with perturbations at given severity."""
    name = perturb_name(clean['name'], severity)
    phone = perturb_phone(clean['phone'], severity)
    website = perturb_website(clean['website'], severity)
    address = perturb_address(clean['address'], severity)
    cat_primary, cat_alts = perturb_category(clean['category_primary'], clean['category_alternates'], severity)

    # Keep confidence only weakly correlated with quality to avoid leakage.
    confidence = clamp(random.gauss(0.68 - 0.10 * severity, 0.18), 0.05, 0.99)

    return {
        'names': json.dumps({'primary': name}),
        'phones': json.dumps([phone] if phone else []),
        'websites': json.dumps([website] if website else []),
        'addresses': json.dumps([address] if address else []),
        'categories': json.dumps({'primary': cat_primary, 'alternate': cat_alts}),
        'confidence': confidence,
    }


def choose_pair_profile():
    """Choose realistic quality profiles for current/base sides."""
    profile = weighted_choice([
        ('current_better', 0.35),
        ('base_better', 0.35),
        ('near_equal', 0.15),
        ('both_noisy', 0.15),
    ])

    if profile == 'current_better':
        return profile, random.uniform(0.02, 0.20), random.uniform(0.35, 0.80)
    if profile == 'base_better':
        return profile, random.uniform(0.35, 0.80), random.uniform(0.02, 0.20)
    if profile == 'near_equal':
        center = random.uniform(0.10, 0.45)
        jitter = random.uniform(0.0, 0.08)
        return profile, clamp(center + jitter, 0.0, 1.0), clamp(center - jitter, 0.0, 1.0)

    # both_noisy
    return profile, random.uniform(0.45, 0.90), random.uniform(0.45, 0.90)


def label_from_quality(curr_sev, base_sev):
    """Assign label from side quality, independent from confidence fields."""
    # Lower severity means cleaner value and should generally win.
    margin = base_sev - curr_sev

    if abs(margin) < 0.04:
        return random.choice(['c', 'b'])

    label = 'c' if margin > 0 else 'b'

    # Inject a small amount of label noise to avoid overfitting synthetic artifacts.
    if random.random() < 0.08:
        return 'b' if label == 'c' else 'c'
    return label


def label_from_confidence(curr_conf, base_conf):
    """Assign label from confidence with a small tie zone."""
    if abs(curr_conf - base_conf) < 0.03:
        return random.choice(['c', 'b'])
    return 'c' if curr_conf > base_conf else 'b'


def create_synthetic_dataset(limit=2000, seed=42, label_mode='quality'):
    random.seed(seed)
    print(f"Generating synthetic records from Yelp (limit={limit}, seed={seed}, label_mode={label_mode})...")

    with open(YELP_PATH, 'r', encoding='utf-8') as f:
        yelp_data = [json.loads(line) for line in f]

    if limit > 0 and limit < len(yelp_data):
        sampled_yelp = random.sample(yelp_data, limit)
    else:
        sampled_yelp = yelp_data
        print(f"Using all {len(sampled_yelp)} Yelp records.")

    records = []
    profile_counts = {
        'current_better': 0,
        'base_better': 0,
        'near_equal': 0,
        'both_noisy': 0,
    }

    for y in sampled_yelp:
        phone_val = y.get('phone') or random_phone_us()
        website_val = y.get('website') or random_website(y.get('name', 'business'))
        categories = normalize_categories(y.get('categories'))

        clean = {
            'name': y.get('name', ''),
            'phone': phone_val,
            'website': website_val,
            'address': {
                'freeform': y.get('address', ''),
                'locality': y.get('city', ''),
                'region': y.get('state', ''),
                'postcode': y.get('postal_code', ''),
                'country': 'US',
            },
            'category_primary': categories[0] if categories else 'business',
            'category_alternates': categories[1:3],
        }

        profile, curr_sev, base_sev = choose_pair_profile()
        profile_counts[profile] += 1

        current = make_side(clean, curr_sev)
        base = make_side(clean, base_sev)
        if label_mode == 'confidence':
            label = label_from_confidence(current['confidence'], base['confidence'])
        else:
            label = label_from_quality(curr_sev, base_sev)

        records.append({
            'id': f"synthetic_yelp_{y['business_id']}",
            'data': {
                'current': current,
                'base': base,
            },
            'label': label,
            'method': f'synthetic_yelp_realistic_v2_1_{label_mode}',
            'profile': profile,
        })

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(records)} synthetic records to {OUTPUT_PATH}")
    print("Profile mix:", profile_counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic golden dataset')
    parser.add_argument('--limit', type=int, default=2000, help='Number of records to generate (0 for all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible generation')
    parser.add_argument(
        '--label-mode',
        choices=['quality', 'confidence'],
        default='quality',
        help='How to derive labels: quality-based or confidence-based.',
    )
    args = parser.parse_args()

    create_synthetic_dataset(args.limit, args.seed, args.label_mode)
