Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 2 items.

=== ITEM id=b57800fa5180d426 ===
Gold sentences and facts:
Sentence 0: Microsoft also has a stripped-down Elite Series 2 “Core” version for $129.99, which is the same controller in white-and-black, red-and-black, or blue-and-black finishes without the rear paddles, accessory charging case, or extra stick toppers and D-pad.
  Fact 0.0: Microsoft also sells a stripped-down Elite Series 2 'Core' version of its controller.
  Fact 0.1: The Elite Series 2 Core version costs $129.99.
  Fact 0.2: The Elite Series 2 Core is the same controller as the full Elite Series 2.
  Fact 0.3: The Elite Series 2 Core comes in white-and-black, red-and-black, or blue-and-black finishes.
  Fact 0.4: The Elite Series 2 Core does not include the rear paddles.
  Fact 0.5: The Elite Series 2 Core does not include the accessory charging case.
  Fact 0.6: The Elite Series 2 Core does not include the extra stick toppers.
  Fact 0.7: The Elite Series 2 Core does not include the extra D-pad.
Sentence 1: While these models are still quite new, versions that actual humans buy have begun receiving discounts.
  Fact 1.0: The models mentioned are still quite new.
  Fact 1.1: Versions that actual humans buy have begun receiving discounts.

Retrieved items:
<item 1>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
However, it starts with just 8GB of RAM and lacks the third USB-C / Thunderbolt port found on the M3 Pro and M3 Max models (as well as Thunderbolt 4 speeds).

The M3 MacBook Pro may be a bit of an odd middle child in some ways, but it’s still a very good laptop — especially if you can get it on a deal that puts more
price distance between it and the M3 Pro version. We’ve already seen the base model with 512GB of storage on sale for $150 off during Black Friday and Cyber
Monday, and right now, it’s actually available for an even better $1,399 ($200 off) at Amazon, Best Buy, and B&H Photo. The 1TB configuration is also selling
for its all-time low of $1,599 ($200 off) at Best Buy and B&H Photo, while the configuration with base storage and 16GB of RAM is selling for $1,599 ($200 off)
at B&H Photo.
</item 1>

<item 2>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
Gurman suggests we'll see USB-C versions of the Magic Trackpad, Magic Mouse and Magic Keyboard at the event.

This makes a lot of sense. Apple ditched the Lightning port in all iPhone 15 models, while it started adopting USB-C on iPads several years back. The company
also recently unveiled an Apple Pencil that has a USB-C charging port. This change was prompted by European Union rules that require a standard charging port on
certain devices. Having the same port on all Apple devices and accessories will mean consumers have fewer types of cables to worry about.

iPad Pro

Nathan Ingraham / Engadget

While Gurman may have poured cold water on the idea, we can't disregard our hunch that we'll see at least one iPad at Apple's event. With the focus likely to be
on M3-powered products, the chances are that any new iPad will be of the Pro variety with one of those chips inside.
</item 2>

<item 3>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The redesigned M2 version of the MacBook Air has been with us for a bit — there’s now even a 15-inch M2 Air — but the 2020 version with an M1 processor and
fanless design remains in the lineup as the budget option. For many people, the M1 Air still ticks the right boxes when it comes to performance and price, even
if it’s long enough in the tooth to have been fully dethroned in our guide to the best laptops.

The base MacBook Air with the M1 chip comes with 8GB of RAM and 256GB of storage. It typically sells for $999, but Amazon and Best Buy are currently discounting
it to $749.99 ($250 off), matching its lowest price. The M1 Air may be a few years old now but it’s still hard to beat in terms of value — especially when it
comes to everyday performance and battery life.
</item 3>

<item 4>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
I’ve begun thinking of the 16-inch MacBook Pro as something more along the lines of a portable desktop. That is to say that the M3 Max is plenty powerful for
even most professional uses. The chip also affords the laptop the ability to power up to four external displays, making it a fine workstation option for many.
When it’s time to take your work home (assuming you don’t live in France), you can pop it in your bag and take it with you. I would recommend against taking
your work (and large laptop) home with you every night, but you do you.

https://techcrunch.com/2023/11/06/apple-imac-m3-review/

The new MacBook Pro, which goes on sale this week, was announced at last week’s Scary Fast event, alongside a new iMac and — of course — several members of the
M3 line. That latter bit marked a big departure for the company, following rumors that Apple had planned the initial M3 launch for WWDC.
</item 4>

<item 5>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
Apple fans love a new color. In the case of the MacBook line, it could be attributable to the fact that there’s little variation in the laptops’ aluminum
design. The last few generations have traded almost exclusively in space gray and silver (dark and light gray, essentially). The new Airs opened this up a bit,
however, with the addition of starlight (a subtle light-gold) and midnight (a much darker gray). The space black is a nice addition. It gives off a nice glow
when the light hits it. The biggest thing here, however, is the fingerprint factor. The Airs I’ve tested were silver and starlight, so I can’t really speak to
this directly, but the midnight Air is a notorious fingerprint magnet. There are pages and pages of conversations about it on Reddit, including “tips” for a
fingerprint-free midnight Mac. Have you tried typing in gloves? Not easy.
</item 5>

<item 6>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
We recommend the M3 for anyone who wants some of the Pro features but doesn't want to drop almost $2,000 to get it.

This MacBook Pro (9/10, WIRED Recommends) is less than a year old, but we only recommend it if you can find it for under $1,800. It packs all the same features
as the latest version and the M2 Pro is more than capable of handling graphically demanding and intensive tasks. This is also the lowest price we've tracked for
this specific configuration.

This M3 Pro MacBook Pro is similar to the other model that's on sale. It just has a larger screen, a more powerful CPU, and slightly better graphics. Apple's
brand-new M3 chips are available in three different MacBook tiers—the M3, M3 Pro, and M3 Max (9/10, WIRED Recommends). They're all included in our guide to the
Best Laptops.
</item 6>

<item 7>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
That could all start with upgraded chipsets. It's been nearly 17 months since Apple's M2 system on a chip (SoC) debuted. With many manufacturers having an
annual cadence for their chipsets, it may be time for Apple to move onto the next generation. (No, shoving two M2 Max chips together isn't quite the same
thing.) For what it's worth, the M2 series debuted 19 months after the M1. So, if Apple announces the M3 at the Scary Fast event, the latest SoC would be
arriving after a similar period of time.

Apple analyst Ming-Chi Kuo has suggested that "M3 series MacBook Pro" will be the focus of the event, indicating that the company will have more than one new
SoC to reveal. As such, there's a chance that we could see an M3 Pro and maybe even an M3 Max during the Scary Fast showcase.
</item 7>

<item 8>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
But it differs by having a larger, 15.3-inch notched screen with 2880 x 1864 resolution, six speakers instead of four, a 10-core GPU that costs extra on the
13-inch, and a higher starting price of $1,299. Bumping up to a larger, faster 512GB of storage will run you $1,499.

Surprisingly, the 15-inch M2 Air started receiving some discounts even before it shipped, which let early adopters save between $50 and $100. As of right now,
you can get it $300 off by ordering it at Amazon or Best Buy, where it’s currently selling for $999. The expanded 512GB model is also $300 off at Best Buy,
selling for $1,199.

The best MacBook Pro deals

The MacBook Pro line is once again in a transitional phase.
</item 8>

<item 9>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
Back in April 2021, the company finally started bumping its webcams from 720p to 1080p. It was a long-awaited refresh for a company that puts so much emphasis
on its mobile imaging. The new hardware was coupled with improvements brought on by the M1’s on-board digital signal processing. While the camera hardware has
remained the same since, updates to the chips have further tweaked picture quality. As it stands now, the webcam is totally fine, and the addition of portrait
and studio lighting modes further improve its. Portrait mode does still have some trouble with edges. I wear over-ear headphones while podcasting, and it hasn’t
quit figured out what to do with the gap between the band and my head.

Frankly, however, Apple’s biggest enemy here is itself. Continuity Camera is a game changer for many. The system lets a mounted iPhone take over webcam duties,
and there’s really no comparing the two. I never go on a trip without the Belkin MagSafe camera mount.
</item 9>

<item 10>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
However, you can also get a tricked-out M1 Max (10-core CPU / 32-core GPU) model with 64GB of RAM and 2TB of storage for $2,499 ($1,800 off) at B&H Photo. I
know that’s not exactly affordable, but that laptop once sold for $4,299 and still performs well in 2023.

The best Mac Mini deals

The latest Mac Mini comes in a base configuration with Apple’s M2 processor or in a more powerful configuration with the M2 Pro. It’s an excellent upgrade to
one of the most affordable yet performant desktop computers you can get (as long as gaming isn’t your priority). The M2 Mac Mini starts with 8GB of RAM and a
256GB SSD for $599, while the M2 Pro model features a superior processor plus gigabit ethernet, 16GB of RAM, and 512GB of storage for $1,299.

The latter model also features an expanded port selection, from two USB-C ports to four.
</item 10>

<item 11>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
It’s simple math. People can’t get enough of the things. The Pro has three Thunderbolt 4/USB-C ports to the Air’s two. Also, for some reason, all of the ports
(save for the headphone jack) are all on the same side with the Air. Things can get crowded fast. Both systems have the latest MagSafe connector — though I’d
like to point out that despite being a bespoke charger, you’re able to top the laptops up faster via Thunderbolt. The other important addition on the Pro side
is an HDMI port, which is key for the aforementioned external display support.

From an industrial design perspective, not much has changed here from the 2021 model. Apple settled into a look it likes, balancing rounded edges with a flat,
almost boxy design that isn’t dissimilar from the iPhone in that respect. On top is the familiar reflective Apple logo, while “MacBook Pro” is etched into the
bottom.
</item 11>

<item 12>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
iMac

A new 24-inch iMac could make its debut at the Scary Fast event. Bloomberg's Mark Gurman reported that a fresh model is expected to arrive this month. He noted
that the iMac is in short supply at Apple retail stores, which is another sign that an upgrade is imminent.

The 24-inch iMac is well overdue for a refresh, and there have been rumblings for a while that an M3 variant was in the pipeline. The most recent model was an
M1-powered machine that arrived in April 2021. Making the leap to M3 — assuming that the new chip lineup is more capable than the M2 series — may make any new
iMac worth the upgrade for many users.

MacBook Pro

Devindra Hardawar/Engadget

There's a very good chance Apple will have at least one MacBook Pro refresh to show off at the event. Both Kuo and Gurman have indicated that will be the case.
</item 12>

<item 13>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
I never go on a trip without the Belkin MagSafe camera mount.

The on-board speaker and microphone systems, meanwhile, remain mostly unchanged from the 2021 M1 Pro. There are six speakers, with two large grilles on either
side of the keyboard. The system supports both Spatial Audio and Dolby Atmos. It’s not an exceptional speaker system, but it gets the job done. The Pro also has
a three-mic array with directional beamforming.

M3 Max

The M3 line is the first of Apple’s desktop chips to be created using the 3nm process, following the iPhone 15 Pro’s lead. Apple has employed the 5nm process
since the 2020’s M1 introduction. The new process brings performance gains with less power consumption, and the Mac is the first PC line to employ the
technology.

Gaming continues to be an important narrative for Apple’s silicon journey, and it was very much at the heart of last week’s M3 announcements.
</item 13>

<item 14>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
Have you tried typing in gloves? Not easy.

Apple describes the new color in the most Apple way possible, “The finish features a breakthrough chemistry that forms an anodization seal to greatly reduce
fingerprints.” The company hasn’t perfected the wholly fingerprint-proof surface, but the new finish does an excellent job keeping things to a minimum.

Camera

Both new Pro models are built around Apple’s Liquid Retina XDR display. The 14-inch sports 3024 x 1964 pixels to the 16’s 3456 x 2234 pixels — both come out to
a pixel density of 254 ppi. The 15-inch Air, meanwhile, has a 224 ppi density. The 16-inch system’s display is largely unchanged from the 2021 model. It has a
1000 nit sustained brightness and 1600 nit peak with HDR (high dynamic range) content. For SDR (standard dynamic range) content, however, brightness has been
bumped up from a 500 to 600 nits max.
</item 14>

<item 15>
The best Xbox controller to buy right now (The Verge, 2023-11-21T20:04:33+00:00)
You can get all of those in a $59.99 accessories pack, which turns the Elite Series 2 Core back into a standard Elite Series 2, but that sets you back $10 more
than just going for the Elite Series 2 to start with.

You must also be aware that the Elite has a bit of a reputation for lackluster quality control, with horror stories from users about going through multiple
replacements under warranty. Microsoft extended the controller’s warranty from 90 days to a year in late 2020 to help address concerns, but buying an Elite
Series 2 may still feel like a slightly risky proposition. However, when you nail that perfect sequence — switching to your pistol and landing a headshot with a
quick pop-pop of the hair trigger while your thumbs stay firmly planted on the sticks — it can feel worth it.
</item 15>

<item 16>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
However, this version is fairly bare-bones—it doesn't come with a Touch ID button or the same function row keys as the M-series MacBooks (that model isn't on
sale, sadly). But it does come with the standard keys like playback controls, a Mission Control key, as well as a number pad. This is also the lowest price
we've tracked for this keyboard, so far.

Apple Studio Display Photograph: Apple

The Studio Display (9/10, WIRED Recommends) has a spacious 27-inch display that's perfect for juggling multiple apps at the same time and a 5K resolution that's
stunningly sharp (there's no HDR though, so colors look slightly more contrasty and saturated compared to the neutral tones on Apple's Pro Display XDR).
Although it's pricey, this is an excellent monitor for those who want a really accurate and sharp screen. Equipped with a 12-megapixel camera, along with
built-in mics and speakers, it's ideal for video calls too.
</item 16>

<item 17>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
This thing is going to get you through your next flight, no problem.

However, if you’re flying a lot and want a MacBook travel companion, I still recommend the 15-inch Air. It continues to be the best MacBook for the vast
majority of users. The 14-inch Pro is worth looking at gaming is important — though I’m still a long way off from recommending any Macs are pure gaming
machines. The 16-inch is the pick for those who really need to up their professional game, but still need some portability. For pure desktop power, the Mac
Studio and Mac Pro with the M2 Ultra is still king — until the M3 Ultra rolls out.

To state the obvious, the 16-inch MacBook Pro with M3 Max is the most powerful Apple laptop, ever. You’ll still get more performance out of the M2 Ultra in the
Mac Studio, but for most intents and purposes, this is the desktop of laptops. It’s big, heavy, expensive and powerful.
</item 17>

<item 18>
The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more (Engadget, 2023-11-27T10:00:30+00:00)
Read our review of the new MacBook Pros with the M3 chip

MacBook Air (15-inch, 2023, M2)

The 15-inch MacBook Air arrived in June and added a little more screen real estate to an already great machine. It’s down to $1,049 for Cyber Monday at Amazon
and Best Buy. That’s the same as the all-time low it’s hit a few times these past couple of months. In addition, it's $200 off at B&H Photo. We gave it a 96 in
our review and liked how the bigger screen let you get the most out of the M2 processor.

Now that the M3 MacBook Pros have arrived, we’re guessing Apple will come out with M3-enabled Air models in 2024. Still the M2 has plenty of processing power
and will serve you well as both a casual machine for playing games and watching shows, but can also hold its own with productivity tasks.
</item 18>

<item 19>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
Apple has at least one more trick — or possibly a treat — up its sleeve this year. On October 30, the night before Halloween, the company is hosting another
event. This one is dubbed “Scary Fast,” which is a strong indicator that Apple will have some speedy new hardware to show off. Unusually, Apple is holding this
event in the evening. It starts at 8PM ET.

This year's iPhones and Apple Watches are out in the wild already, but Apple has a bunch of other product categories that it could focus on here. There's always
the chance that we might get iPad updates. But, if rumors are to be believed, Apple's focus during the Scary Fast event will be on Macs and a fresh batch of
chipsets to power them.

M3 chip

Apple won't be talking about speed advancements without having new hardware to discuss. That could all start with upgraded chipsets.
</item 19>

<item 20>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
Unless demand drops, Kuo said supply may remain tight until early 2024.

MacBook sales have waned this year, mainly due to "the limited M2 computing power upgrade," Kuo wrote on X. That might be a reason why Apple is willing to start
shipping M3 MacBook Pros now, even if supply is constrained. Announcing new models just a few days before reporting quarterly earnings could help give investors
confidence in the company (recent increases of Apple TV+, Apple Arcade and Apple News+ prices could play into that too).

If M3-powered models don't lead to a boost in sales, though, Kuo suggests it's more likely that a MacBook Pro with a new design will arrive in 2025.
Furthermore, Apple may try to beef up the division with a more budget-friendly MacBook that has an annual sales target of between 8 million and 10 million.

USB-C accessories

Rumors also point to Apple unveiling more USB-C accessories as it shifts away from the Lightning port.
</item 20>

<item 21>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
GPU updates were the centerpiece of the news for several reasons. One is a feature the company calls Dynamic Caching. Apple writes,

With Dynamic Caching, only the exact amount of memory needed is used for each task. This is an industry first, transparent to developers, and the cornerstone of
the new GPU architecture. It dramatically increases the average utilization of the GPU, which significantly increases performance for the most demanding pro
apps and games.

More simply put, the idea is allocating just the right amount of local memory to a given task. It’s something Apple has been working on for a while. The other
two additions should be more familiar for anyone who’s spent time behind a game controller. The first is ray tracing, a feature designed to bring added realism
to the gaming experience, though more accurate depictions of lighting and reflections.

That feature is now hardware-accelerated, along with mesh shading. The triangle computer shader makes processing complex geometrical images more efficient.
</item 21>

<item 22>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
You can see a pattern emerging. The M3 Max outperforms the M3 on everything (no surprise there) and the M2 Ultra on several single-core tests, while often
coming within spitting distance with some of the multi-core tests.

With GeekBench 6 Metal, which tests GPU performance during complex tasks, the M3 Max scored an average of 153273. That’s a nice increase over the M2 Max’s
131883 average, while still getting trounced by the M2 Ultra’s 208584. AMD’s cards still best even the top-performing Apple chips by a good margin, with the
Radeon RX 6900XT scoring 242924.

Relative to its own family of chips, the M3 Max is doing impressive things, given that the M2 Ultra tops out at a 24-core CPU, 76-core GPU and 32-core neural
engine. The Mac Studio’s large body and airflow system are also much more equipped to dissipate heat, when compared to the MacBook Pro’s compact profile.
</item 22>

<item 23>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
It’s certainly a system that’s likely to last you a while. It’s true that Apple is currently on a one-year upgrade cycle for it first-party M-series chips, but
it’s genuinely tough to imagine and advanced as pronounced as the one it made from Intel to in-house silicon happening any time soon. There’s no such thing as a
future-proofed laptop, so the best you can hope for is one that will give you good mileage. That certainly applies here.

Along with building new chips, Apple has spent the last few years listening to consumers in a way it hadn’t for decades. That means finally updating the camera,
building a better keyboard (the class action settlement probably tipped the scales as well) and ditching the well-meaning but ultimately ineffectual Touch Bar
altogether.
</item 23>

<item 24>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
You can, of course, spec them up the wazoo if you’re willing to pay more, as exhibited by the review unit Apple sent us of the 16-inch model, which cost an
eye-watering $7,199.

While these models are still quite new, versions that actual humans buy have begun receiving discounts. The 14-inch MacBook Pro in its base configuration with
the M3 Pro (11-core CPU / 14-core GPU), 18GB of RAM, and 512GB SSD is currently on sale for $1,749 ($250 off) at B&H Photo. A similar version with 1TB of
storage and a 12-core / 18-core M3 Pro is down to $2,149 (also $250 off) at B&H Photo. As for the base 16-inch MacBook Pro with the M3 Pro (12-core CPU /
18-core GPU), 18GB of RAM, and a 512GB SSD, it’s selling for $2,249 ($250 off) at Amazon and B&H Photo.
</item 24>

<item 25>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
Both Kuo and Gurman have indicated that will be the case.

Gurman pointed out that the 13-inch, 14-inch and 16-inch MacBook Pro all currently have extended shipping times on Apple's web store. That's often an indicator
that the company is winding down the supply of existing products. Furthermore, Gurman reported that Apple has been testing M3 Max and M3 Pro MacBook Pro models.
Those could now be ready for prime time.

However, Kuo doesn't expect Apple to have a huge inventory of new MacBook Pro units to sell this year. He previously didn't anticipate that Apple would start
selling M3-powered laptops before the end of the year due to limited supply of between 400,000 and 500,000 units. Unless demand drops, Kuo said supply may
remain tight until early 2024.
</item 25>

<item 26>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
It's $200 off at B&H Photo. We gave it a 96 in our review and liked how the bigger screen let you get the most out of the M2 processor.

Now that the M3 MacBook Pros have arrived, we’re guessing Apple will come out with M3-enabled Air models in 2024. Still the M2 has plenty of processing power
and will serve you well as both a casual machine for playing games and watching shows, but can also hold its own with productivity tasks.

MacBook Air (13-inch, 2022, M2)

The M2-powered, 13-inch MacBook Air is on sale for $929 at B&H Photo. It’s our top recommendation for a MacBook and beat out all brands in our laptop guide. The
M2 chip is impressively fast and capable, the design is solid and the display is lovely. It’s more than capable for both work and play and should give you years
of dependable use.
</item 26>

<item 27>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The supply chain ultimately thought different(ly). What announcing the M3, M3 Pro and M3 Max at once affords the company, however, is options — though less so
for the M3-only iMac.

Between the 14- and 16-inch MacBook Pros, you’ve got all three configurations covered. Presumably an M3 Ultra is also on the way in the not-too-distant future,
but will almost certainly be confined to desktops like the Mac Studio, and Mac Pro, owing to the limitations of system architectures. Apple also recently shot
down rumors surrounding a 27-inch iMac. The 14-inch MacBook Pro can be configured with any of the three new chips, while the 16-inch is limited to just the Pro
and Max.

Apple sent us the 16-inch with the M3 Max. The company, understandably, likes to put its best foot forward with this stuff (so yes, it’s also the Space Black
model).
</item 27>

<item 28>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The Max comes in both 14-core CPU/30-core GPU and 16-core CPU/40-core GPU flavors. Again, the one I am typing this on falls into the latter camp. That also
means 48GB of memory and 1TB of storage by default. It’s a $4,000 computer, as configured. If you want to bump the memory and storage up to 128GB and 8TB,
respectively, it comes to $7,200. That’s a pretty hefty increase from the base level 14- and 16-inch models, which run $1,600 and $2,500 to start.

Apple lowered the barrier of entry for the lowest end model, but — as ever — upgrades start to add up quickly. We’re talking about a real investment here for
you — or better yet, your company. It’s certainly a system that’s likely to last you a while.
</item 28>

<item 29>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
This thing is a beast — in pretty much every sense. It’s a big, heavy, powerful system, with a bright display, massive battery life and impressive power under
the hood. After opening the box, I immediately ported all the contents over from the 15-inch Air I’ve been using daily for several months now, but still, the
16-inch MacBook Pro feels like a different species.

Each has their place. I’m hopping on another cross-country flight tomorrow, and I’m planning to leave the 16-inch Pro at home. This is due partially to an
ongoing back injury, but frankly, for as good as the Pro’s battery life is (very good, for the record), the Air is a much better travel companion. It’s thin,
it’s light, it has more than enough processing power for 95% of the things 95% of people need, and it won’t wrestle you for the arm rest.
</item 29>

<item 30>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The Ultra, for example, is for all in intents and purposes, two Maxes stitched together. There are other factors at play as well, like the fact that Mac Studio
and Mac Pro do a better job with heat dissipation, while the MacBook Pros are more equipped than the fan-less Airs, which throttle performance more quickly due
to heat.

There are still a handful of ways, however, that even the baseline M3 will outperform the M2 Ultra. Two clear instances are ray tracing and mesh shading, which
were previously software-accelerated.

With Blackmagic Disk Speed, which measures SSD performance, the 16-inch Pro scored 6367 write and 5057 read speeds, vs. the Mac Studio’s 6648/5716. Running
Geekbench 6, the system scored 3149 on the single-core test and 21269 multi-core. That’s a nice bump over the M3 iMac’s 3024/11810.
</item 30>

<item 31>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
There were still a few during the event, but four-year-old non-proprietary chips are no longer the main point of comparison. Enough time and upgrades have
elapsed since the M1 arrived. In fact, this Friday marks three years since it was announced. Apple again,

Rendering speeds are now up to 2.5x faster than on the M1 family of chips. The CPU performance cores and efficiency cores are 30 percent and 50 percent faster
than those in M1, respectively, and the Neural Engine is 60 percent faster than the Neural Engine in the M1 family of chips.

It’s worth pointing out here that Apple’s chip numbering system isn’t as straight forward as it seems at first blush. For instance, while the M3 represents an
advance over the M2, you can still expect to see, say, the M2 Ultra to outperform the M3. Pros, Maxes and Ultras are progressively larger versions of their
baseline chip, containing more CPU, GPU and neural engine cores.
</item 31>

<item 32>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
In fact, it shouldn’t go unremarked upon that the arrival of the new 14-inch signaled the merciful end to that particular technology, as the last Touch Bar Mac
— the 13-inch Pro — was replaced by the new 14-inch model.

Design

I would say, more than anything, the thing I miss moving from the MacBook Pro to the Air is the ports. True story: At last year’s CES, I had to locate an Apple
Store in the Venetian because my external SD reader failed (I won’t say which TechCrunch staff member laughed at me in that predicament, but they know who they
are). I understand that an SD reader is kind of a niche need in these days of everyone shooting everything on their, so I can at least understand Apple’s
justification for keeping it to the Pro system (the new iMac doesn’t have one, either).

Ports are a different story. More ports is better. It’s simple math. People can’t get enough of the things.
</item 32>

<item 33>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
Around that, you’ll see a series of small screws that signal the company’s newfound push into repairability spurred on by the broader call for right to repair
legislation. There is also a heat vent on either side.

The Pros are a good bit thicker than the Air. The 14- and 16-inch are 0.61 and 0.66 inches tall, respectively, versus the 15-inch Air’s 0.45 inches. This is
due, in no small part to the Pros’ inclusion of a fan and airflow system. You’re unlikely to trigger the fan during most everyday operations, but more complex
tasks like 3D rendering require a lot of compute power, so it’s there when you need it. Meanwhile, the 14-inch Pro (3.4 pounds) weighs just a hair more than the
15-inch Air (3.3 pounds). At 4.7/4.8 pounds (for the M3 Pro and Max), there’s a pronounced increase with the 16-inch Pro.
</item 33>

<item 34>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Microsoft Surface Pro 9

A configuration of Microsoft's Surface Pro 9 with an Intel Core i5-1235U processor, 16GB of RAM, a 256GB SSD and Microsoft's Surface Pro Keyboard is on sale for
$1,000 at Best Buy. That's $540 off Microsoft's list price. If you don't need the keyboard, a variant with a faster Core i7-1255U chip is down to a new low of
$1,097 at Amazon, Microsoft, B&H and other retailers. That's roughly $200 off the config's usual street price. Another Core i7 model with 512GB of storage is on
sale for $300 more.

In any event, the Surface Pro 9 tops our guide to the best 2-in-1 tablets. While its 12th-gen chip is about to be two generations old, this is still the device
to get if you want the functionality of a laptop in the slim, touch-enabled design of a tablet.
</item 34>

<item 35>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
But if you want to buy new and you’re looking to save whatever you can, here are the best MacBook Air, MacBook Pro, and Mac Mini deals available right now.

A note about discounted Intel-based Macs At this point, we don’t recommend buying any older Intel-based Macs. Retailers may try to tempt you with very low
prices on leftover new or refurbished stock, but there’s a reason nobody wants them anymore. They’re completely outclassed by all the newer models, unless you
have some super-niche Windows Boot Camp needs — and even then, consider what Windows apps you really need and if they work with Parallels.

The best MacBook Air deals

M1 MacBook Air

The MacBook Air is Apple’s entry-level laptop. It’s best suited for typical productivity work, with a comfortable keyboard, an excellent trackpad, and all-day
battery life.
</item 35>

<item 36>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
M2 MacBook Air

As for the newer, fancier 13-inch MacBook Air with M2, the base model with 256GB of storage and an eight-core GPU recently got a price drop. Along with
introducing a 15-inch M2 Air, Apple announced during this year’s WWDC that the 13-inch model now starts at $1,099 instead of $1,199. As for deals, the 13-incher
is currently selling for $899 ($200 off) at B&H Photo, Additionally, Best Buy and B&H Photo are offering an equal diacount on the upgraded configuration, which
can be had with a 512GB SSD and 10-core GPU for $1,199 ($200 off).

The M2 MacBook Air is a super slim and light laptop with a 1080p webcam that’s actually usable as well as a handy magnetic charger that frees up one of its
precious two USB-C ports.
</item 36>

<item 37>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
If it's within your budget, we recommend the model with 16GB of RAM, which is $150 off at B&H photo. And if you upgrade to a model with the M3 Pro chip for an
even speedier experience — you'll get a $200 discount as it's down to $2,199 at B&H Photo. We gave the new M3 MacBook Pros an appreciative review and named them
the best option for creatives in our buyer’s guide.

Read our review of the new MacBook Pros with the M3 chip

MacBook Air (15-inch, 2023, M2)

The 15-inch MacBook Air arrived in June and added a little more screen real estate to an already great machine. It’s down to $1,049 for Black Friday at Amazon
and Best Buy. That’s the same as the all-time low it’s hit a few times these past couple of months. It's $200 off at B&H Photo.
</item 37>

<item 38>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The best MacBook Pro deals

The MacBook Pro line is once again in a transitional phase. Apple announced Macs using a new M3 processor at its “Scary Fast” event the night before Halloween,
including a new 14-inch MacBook Pro that replaces the 13-inch M2 model. That means Touch Bar Macs are finally dead, but some of those outgoing M2 models are
still available for now. You’ll find any deals on them further down, but first, let’s focus on the new M3 generation.

M3 and M3 Pro MacBook Pro

The new entry point into the MacBook Pro world is a MagSafe-equipped MacBook Pro that uses the existing 14-inch design and slightly pares it down. The 14-inch
M3 MacBook Pro has the same 3024 x 1964 resolution display and 120Hz refresh rate as its pricier siblings, and in addition to a MagSafe charging port, it has an
SD card slot and HDMI port.
</item 38>

<item 39>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
M2 Pro MacBook Pro

As for the 14- and 16-inch MacBook Pros from early 2023 with M2 Pro / Max processors, they’re still more than capable — they may even be a better buy than the
new models if you can find them at a steep discount.

For example, the base 14-inch MacBook Pro with an M2 Pro chip, a 512GB SSD, and 16GB of RAM is selling for $1,629 ($370 off) at Amazon, which is a new low price
that may not last very long. However, the 14-inch model with 1TB of storage, a 12-core M2 Pro CPU, and 19 GPU cores is down to $1,999 (a steep $500 off) at B&H
Photo. Now we’re talking!

Sadly, the situation isn’t quite as good for the 16-inch MacBook Pro with the M2 Pro.
</item 39>

<item 40>
The best Xbox controller to buy right now (The Verge, 2023-11-21T20:04:33+00:00)
However, using that many buttons on the back of the controller definitely takes some adjustment. You have to train new muscle memory, and since it requires
dexterity with more fingers, they’re not for everyone.

These premium features come at a steep cost, too. The Elite Series 2 is normally priced at $179.99, and even though it occasionally drops to $139.99, it’s still
more than twice as expensive as a regular Xbox controller. You have to really want its extra features and user customization, and if you’re not using its rear
paddles or the software adjustments for stick sensitivity, you’re not getting your money’s worth. (And if you want to customize the colors, you’re looking at
$209.99 for the Design Lab “Elite Package.”)

Microsoft also has a stripped-down Elite Series 2 “Core” version for $129.99, which is the same controller in white-and-black, red-and-black, or blue-and-black
finishes without the rear paddles, accessory charging case, or extra stick toppers and D-pad.
</item 40>

<item 41>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
That said, if you plan on going with a higher-spec configuration than that, you’ll likely want to spring for the M3 Pro model instead (see below).

The 14- and 16-inch MacBook Pros for late 2023 are another round of spec-bump models, much like their previous 2022 incarnations. Now starting with the M3 Pro
chip (or the speedier M3 Max) and 18GB of base RAM instead of 16GB, the new models remain targeted at creatives doing content work like video editing, photo
processing, and other graphical work. Like their predecessors and the M1 Pro generation before that, they offer MagSafe charging, three USB-C / Thunderbolt 4
ports, HDMI-out, and a full-size SD card slot, with prices starting at $1,999 for the 14-inch and $2,499 for the 16-inch.
</item 41>

<item 42>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
Apple sells MacBooks equipped with its own M-series chips in a wide range of sizes and price points. The offerings start with the 13-inch MacBook Air from 2020
at $999 and go all the way up to the latest 16-inch MacBook Pro starting at $2,499. But finding a deal on a current Mac with an M1, M2, or even the new M3 chip
— as well as the higher-end M3 Pro and M3 Max — is actually not that difficult.

While Macs may not experience perpetual discounts, it’s not uncommon to see various current models discounted by as much as $400. Alternatively, purchasing
refurbished options directly from Apple is another way to save money without as much waiting for the changing winds of deals to blow your way. Apple’s
refurbished store provides a one-year warranty on all products and generally offers discounts of up to 15 to 20 percent off the price of a new unit.
</item 42>

<item 43>
The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more (Engadget, 2023-11-27T10:00:30+00:00)
Check out our guide to iPads here.

MacBook Cyber Monday deals

MacBook Pro (14-inch, 2023, M3)

The base model, 14-inch MacBook Pro with the new M3 chip just came out in October but is $150 off for Cyber Monday. After testing the new chip, we think the
base model with 8GB of RAM may quickly get bogged down. If it's within your budget, we recommend the model with 16GB of RAM, which is $150 off at B&H photo. And
if you upgrade to a model with the M3 Pro chip for an even speedier experience — you'll get a $200 discount as it's down to $2,199 at B&H Photo. We gave the new
M3 MacBook Pros an appreciative review and named them the best option for creatives in our buyer’s guide.
</item 43>

<item 44>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
This helps if you’re working on pro-level tasks like video editing but can’t spend the premium that Apple charges for its bigger Pro models.

The Razer Blade 14 earned an 8/10, WIRED Recommends award in our review. It has a gorgeous 16:10 display, plenty of USB ports, a full-size HDMI output, and an
Nvidia RTX 3080 Ti, with the option to upgrade to the 40-series, giving it plenty of power to tear through the toughest games. It's the laptop WIRED reviewer
Eric Ravenscraft used through Starfield's launch weekend; it has proven its worth already.

Photograph: Das Keyboard

You may have to click on “See More Buying Options” to see this deal. The Das Keyboard MacTigr pairs well with Apple's laptops and PCs. It has a dedicated Mac
layout, Cherry MX Red switches, a two-port USB-C hub, and a high-quality, all-metal build. It rarely goes on sale.
</item 44>

<item 45>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The triangle computer shader makes processing complex geometrical images more efficient. That, in turn, allows for higher-quality graphics and more scene
detail.

The results are impressive. I’ve been spending a bit of time with the Soulslike title, Lies of P. It’s basically Pinocchio with more swords. It’s of a small –
but increasing – number of day-and-date releases for macOS. A couple of years ago, it would have been inconceivable for the operating system to get the game the
same day as PS5, Xbox Series X/S and Windows. That said, the macOS gaming library still has a long way to go.

Indeed, the gameplay was smooth and detailed – something that also would have seemed impossible not all that long ago. Sony’s PS5 DualShock controller connects
seamlessly and makes for a fun portable gaming experience.

One thing I noticed during last week’s presentation is that Apple is leaning less heavily on Intel comparisons.
</item 45>

<item 46>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
That’s a nice bump over the M3 iMac’s 3024/11810. In our Mac Studio review, the M2 Ultra got 2819 single and 21507 multi-core. Taking a step back in time to
Geekbench 5 (which we’ve been running on M-series systems from the outset), the M3 Max gets a single-core score of 2323 and 23190 on the multi-core. That’s a
big performance increase over the M3 iMac’s 2191/10619 and the M2 Max 14-inch Pro‘s 1952/15249.

The M3 Max powering our Pro sports 16 CPU cores — that’s 12 performance and 4 efficiency — while the GPU packs in a whopping 40 cores. On Cinebench 2024, it
scored 142 for single-core and 1693 with the multi-core, compared to the M3 Max’s 138/617. You can see a pattern emerging.
</item 46>

<item 47>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The best (and only) deal available right now on the creator-friendly laptop is at Amazon, where you can grab the base M2 Pro model for $2,299.99 ($200 off) with
512GB storage. That’s not great, but hopefully we see better deals before the laptop is no longer available new.

M1 Pro and M1 Max MacBook Pro

For now, the older 2021 MacBook Pros can occasionally be had for some steep discounts that actually make a two-generation-old laptop still worth considering.
You’re unlikely to find any base models still kicking around, but if you want a souped-up configuration on the cheap, you may be in luck.

The 16-inch MacBook Pro from 2021 with 16GB of RAM and an expanded 1TB of storage is available at B&H Photo for $1,799 ($900 off).
</item 47>

<item 48>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The latter model also features an expanded port selection, from two USB-C ports to four. It’s almost like getting an M2 Pro-powered MacBook Pro 14 but in
desktop form. However, keep in mind that buying any Mac Mini means you have to provide your own mouse, keyboard, and monitor.

Right now, the base M2 Mac Mini is selling for a new all-time low of $479 ($120 off) at Amazon and B&H Photo, while the M2 Pro version is selling for a
respectable $1,149 ($150 off) at B&H Photo. The M2 Mac Mini remains a great value for a desktop computer, with the M2 Pro version functioning like a baby Mac
Studio for much less.

Related The best thing about my Mac Studio is its mustache
</item 48>

<item 49>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
Its M2 processor didn’t kick-start a revolution like the M1 generation, but it’s a great performer for any user outside of more demanding creatives.

It does have some slight downsides, like slower storage in the base 256GB configuration and a notch cutout in its otherwise excellent screen. But even so, there
hasn’t been a more travel-friendly laptop offered by Apple since the days of the polarizing 12-inch MacBook, and this one’s good enough to be the No. 1 laptop
we now recommend in our buying guide.

The newer 15-inch MacBook Air with M2 is a lot like a blown-up 13-inch Air, and judging from our glowing review, there’s nothing wrong with that. Like its
smaller counterpart, it has an M2 chip, two Thunderbolt 4 / USB-C ports, MagSafe charging, and its base model comes with a 256GB SSD and 8GB of RAM.
</item 49>

<item 50>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
They carry a list price of $169 but often sell for closer to $150, so this is $10 off the usual. These non-Pro AirPods came out in 2021 and got a review score
of 88 from us. They pair quickly with iPhones and are a good complement to Siri’s helpfulness. They don’t have ANC however — you’ll need to go for the AirPods
Pro for that feature — but they do have Adaptive EQ which monitors and adapts sound in real time. The fit is comfortable and they’re water-resistant so they can
handle workouts or the weather.

MacBook Black Friday Day deals

MacBook Pro (14-inch, 2023, M3)

The base model, 14-inch MacBook Pro with the new M3 chip just came out in October but is already$150 off for Black Friday. After testing the new chip, we think
the base model with 8GB of RAM may quickly get bogged down.
</item 50>
=== END ITEM id=b57800fa5180d426 ===

=== ITEM id=67051dcb55929978 ===
Gold sentences and facts:
Sentence 0: Automakers and technology companies are building ever-more sophisticated digital platforms into the future generations of cars and other vehicles.
  Fact 0.0: Automakers are building ever-more sophisticated digital platforms into future generations of cars and other vehicles.
  Fact 0.1: Technology companies are building ever-more sophisticated digital platforms into future generations of cars and other vehicles.
Sentence 1: Now we have software, robotics, AI and all that stuff, and the car companies need to embrace that.
  Fact 1.0: The speaker says the industry now has software, robotics, AI, and similar technology.
  Fact 1.1: Car companies need to embrace software, robotics, AI, and similar technology.

Retrieved items:
<item 1>
SoftBank Corp takes 51% of Cubic Telecom for $513M to drive into the connected car world (TechCrunch, 2023-12-05T07:24:23+00:00)
The bigger gap in the market that the company has been targeting and building to plug for the last several years has been that, while tech companies and
automakers have been working on ways to build more functionality into vehicles — whether that be more clever ways to manage and respond to diagnostics on the
vehicle, or give you an easier way to listen to Spotify, or to help drive a car altogether — the piece that has remained more tricky is the internet
connectivity to make all of that work.

Typically, car makers will have to cut deals with individual carriers region-by-region, which is time-consuming, costly and does nothing for creating a seamless
experience at the front end for users, either. Cubic’s platform acts as a wholesale aggregator, and it essentially helps to manage all of that automatically and
at a lower cost, so that cars come delivered into a market ready to use, and if those cars are then driven into another region, they will continue to work
there, too.
</item 1>

<item 2>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Future of flight

EVTOL company Archer Aviation plans to start air taxi operations in Abu Dhabi in 2026. From there, Archer says it will expand across the UAE as part of a
memorandum of understanding with the Abu Dhabi Investment Office.

Miscellaneous

The United Auto Workers strike is affecting CES. Stellantis canceled its planned presentations for the tech trade show in January, citing the cost of ongoing
UAW strikes.

Zipcar is getting hit with a fine from the NHTSA for “renting vehicles with open, unrepaired recalls.” One recall concerns 2015–2017 Ford Transit Vans, which
continued to appear on Zipcar’s platform despite safety issues with the vehicles.

People

Autobrains hired Uri Yacovy, a former SVP at Mobileye, as its chief operating officer.

Logistics company Flexport is laying off 20% of workers, or about 600 people, topping off a spate of staff upheavals at the company.
</item 2>

<item 3>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
You cannot escape conversations about AI no matter how far or fast you run. Hyperbole abounds around what current AI tech will be able to do (revolutionize
every industry!) and what current AI tech will be able to do (take over the world!). Closer to the ground, TechCrunch+ is working to understand where startups
might find footholds in the market by levering large language models (LLMs), a recent and impactful new method of creating artificially intelligent software.

How AI will play in startup land is not a new topic of conversation. A few years back, one venture firm asked how AI-focused startups would monetize and whether
they would suffer from impaired margins due to costs relating to running models on behalf of customers. That conversation died down, only to come roaring back
in recent quarters as it became clear that while LLM technology is quickly advancing, it’s hardly cheap to run in its present form.

But costs are only one area where we have unanswered questions.
</item 3>

<item 4>
Digital twin market: Analyzing growth and 4 emerging trends (Iot Business News, 2023-11-18T16:01:30+00:00)
By the IoT Analytics team.

A new report from IoT Analytics highlights eight notable trends helping to advance and promote digital twins.

Four of these trends are discussed in detail in this article. These trends are shaping the future of the digital twin market and influencing investment
priorities for companies across various industries.

Key insights:

According to our Digital Twin Market Report 2023–2027, the digital twin market is expanding, with a projected CAGR of 30% between 2023 and 2027.

29% of global manufacturing companies have either fully or partially implemented their digital twin strategies. Further, job posts related to digital twins have
increased by 11% compared to October 2021, while openings for other tech topics have declined in the same timeframe.

The report notes eight notable trends helping to advance and promote digital twins, four of which are discussed in this article.
</item 4>

<item 5>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
I would expect an AI that focuses on practical applications of users’ own data, using Apple’s increasingly central position in their lives to integrate the many
signals and ecosystems the company is privy to. There will likely also be a clever and elegant way to handle problematic or dangerous prompts, and although it
will almost certainly have multimodal understanding (primarily to handle user images), I imagine they’ll totally skip media generation. Expect some narrowly
tailored but impressive agent capabilities as well: “Siri, get a table for 4 at a sushi place downtown around 7 and book a car to take us” sort of thing.

What’s hard to say is whether they will bill it as an improved Siri or as a whole new service, Apple AI, with a name you can choose yourself. They may feel the
old brand is freighted with years of being comparatively incapable, but millions already say “hey Siri” every 10 seconds so it’s more likely they’ll opt to keep
that momentum.
</item 5>

<item 6>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
The former Amazon and Google executive, who has a background in strategy and management consulting, told TechCrunch that when he was struck by the urge to start
a company, he researched the money-attracting industries of the world, and then, using AngelList, saw how many companies were trying to disrupt those
industries.

Here’s the nugget:

His search yielded thousands of companies that were working on industries ranging from telecommunications and fashion to video games and food. Billions of
dollars were going into trucking each year but fewer than 30 startups showed an interest in the field.

“I saw a massive opportunity and few people going after it,” Lewis told TechCrunch.

And then later, when asked if his method of deciding on a direction for a startup is still a good method, Lewis said, in short, yes. Read the whole interview
from May 2022 here.

Other deals that got our attention . . .
</item 6>

<item 7>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Tesla also gave some updates about its long-delayed Cybertruck. Initial deliveries are set for an event at Giga Texas on November 30. Elon Musk noted that
scaling the Cybertruck will be hard and it will take 18 months before the pickup is profitable.

“I mean, we dug our own grave with Cybertruck,” said Musk.

The billionaire executive also said Giga Texas will be able to produce about 250,000 Cybertrucks a year starting in 2025. But let’s remember that Musk isn’t
great at making predictions. After all, he initially said the Cybertruck would be on the market by 2021. Expect some of these numbers to be pushed out, too.

Want to reach out with a tip, comment or complaint? Email Kirsten at kirsten.korosec@techcrunch.com or Rebecca at rebecca.techcrunch@gmail.com.

Reminder that you can drop us a note at tips@techcrunch.com.
</item 7>

<item 8>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
The Station is a weekly newsletter dedicated to all things transportation. Sign up here — just click The Station — to receive the newsletter every weekend in
your inbox. Subscribe for free.

Welcome back to The Station, your central hub for all past, present and future means of moving people and packages from Point A to Point B.

It was an absolutely wild week on the robotaxi front, and more specifically for GM’s self-driving car subsidiary Cruise.

The week started off with the California Department of Motor Vehicles suspending Cruise’s driverless and deployment permits (with the California Public
Utilities Commission following shortly after), effectively ending the company’s robotaxi operations in San Francisco just months after receiving the last
necessary permit to commercialize its operations.

Two days later, Cruise decided to pause driverless operations in every market it had started to charge for its robotaxi service, including Austin, Houston and
Phoenix.
</item 8>

<item 9>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Petersen has spent the past month cutting costs, including laying off about 20% of its workers, or about 600 people.

If Flexport acquires the technology, the company plans to restore Convoy’s trucking services for as many customers and partners as possible, according to WSJ’s
source. And folks, that’s going to cost money. Is Petersen’s reign of financial frugality already over?

Other deals that got my attention …

Faction, the driverless tech developer, raised an undisclosed amount in a round led by TDK Ventures. Ducera Partners, Trucks Venture Capital and Fifty Years
also joined the round.

Ola Electric raised $384.4 million in a funding round, which included about $240 million in debt. Singapore’s sovereign wealth fund Temasek led the funding
round and Indian government-backed lender State Bank of India bankrolled the debt.
</item 9>

<item 10>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
A Google spokesperson said the ability to see live traffic conditions and business information was halted temporarily out of “consideration for the safety of
local communities.” Google did something similar in 2022 amid the Russian invasion of Ukraine because the apps were being used to track military movements.

XPeng, often called the Chinese challenger to Tesla, has removed high-definition mapping in its XNGP assisted driving feature following its rival’s lead.

Ride-hailing and car-sharing

Rapido, the eight-year-old Indian bike taxi startup, is expanding into the cab market in the South Asian nation, where Uber and its homegrown competitor Ola
dominate.

Peer-to-peer car-sharing marketplace Turo has introduced a buy-now-pay-later option. Now you can reserve a car without immediately paying for it until seven
days before a trip. Turo says it built the BNPL option in-house.

UAW strike

Progress was made as the United Autoworkers strike wrapped up its sixth week.
</item 10>

<item 11>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Notable reads and other tidbits

Autonomous vehicles

Waymo driverless vehicles are now available through the Uber app, starting with Phoenix. The launch comes five months since the two companies announced a
multi-year agreement for the autonomous vehicle service to be accessed via the Uber app.

Electric vehicles, batteries & charging

Ford is delaying about $12 billion in planned investments on EVs, including construction of a second battery plant with joint venture partner SK On due to
softening demand for higher-priced premium electric vehicles. While EV sales have grown, consumers aren’t willing to pay a premium for an EV over a gas or
hybrid vehicle. That price pressure has squeezed profits, and in the case of Ford’s EV business caused losses to grow.

General Motors and long-time partner Honda have ended plans to build millions of affordable and smaller electric vehicles as the automakers come to terms with
high interest rates and battery costs coupled with softening EV demand. Anyone spotting a trend here?
</item 11>

<item 12>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
The UAW struck tentative deals with Ford and Stellantis this past week. Workers still have to ratify the deal, but with UAW president Shawn Fain’s support that
outcome is likely.

However, over at GM, it appears that negotiations are moving in the opposite direction. The UAW called for a surprise walkout at GM’s Spring Hill, Tennessee
factory, a plant where 4,000 workers assemble engines and three Cadillac models.
</item 12>

<item 13>
Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation (TechCrunch, 2023-10-28T13:04:25+00:00)
), that it truly needs to feel absolutely confident before it begins implementing new technologies into its workflows.

Another noteworthy piece of news is an Amazon, MIT/Ipsos partnership designed to gauge what both workers and consumers think about industrial robots.

“The key to effective teamwork is building a shared understanding of what our partners will do and what they will need to be successful,” says MIT’s Julie Shah.
“Our research shows that the best way to optimize human-robot team performance is to develop robots that are active collaborators in helping a human to learn
about their capabilities, limitations and behaviors.”

I do think human perception of robots is a question worth asking, but I would love to see a study with such financial and academic resources digging more deeply
into questions around short- and long-term displacements.

During his presentation, Brady addressed the jobs question accordingly:

We have more than 750,000 mobile robots in our operations and thousands of other robotic systems that help move, sort, identify and package customer orders.
It’s taken us more than 10 years to reach this scale.
</item 13>

<item 14>
Robotics Q&A with Boston Dynamics’ Aaron Saunders (TechCrunch, 2023-12-09T16:05:51+00:00)
Following manufacturing and warehouses, what is the next major category for robotics?

Those two industries still stand out when you look at matching up customer needs with the state of art in technology. As we fan out, I think we will move slowly
from environments that have determinism to those with higher levels of uncertainty. Once we see broad adoption in automation-friendly industries like
manufacturing and logistics, the next wave probably happens in areas like construction and healthcare. Sectors like these are compelling opportunities because
they have large workforces and high demand for skilled labor, but the supply is not meeting the need. Combine that with the work environments, which sit between
the highly structured industrial setting and the totally unstructured consumer market, and it could represent a natural next step along the path to general
purpose.

How far out are true general-purpose robots?

There are many hard problems standing between today and truly general-purpose robots. Purpose-built robots have become a commodity in the industrial automation
world, but we are just now seeing the emergence of multi-purpose robots.
</item 14>

<item 15>
Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation (TechCrunch, 2023-10-28T13:04:25+00:00)
The first is the company’s pilots with Agility’s Digit systems. I wrote about this a couple of times last week, including a piece titled “Humanoid robots face a
major test with Amazon’s Digit pilots” that went up over the weekend.

I do genuinely believe there are going to be a lot of eyes on this thing. It’s not that I think it’s the end of Agility if Amazon opts not to extend a contract.
It’s more that if Amazon decides to pursue it further, it’s going to cause a lot more companies to take bipedal/humanoid robots a lot more seriously. I’ve been
saying the whole time that I’m holding off on judging the efficacy of humanoids until we see more in the field, and Amazon clearly feels the same way.

The company operates at such an unfathomable scale (have you visited a regional fulfillment center lately?), that it truly needs to feel absolutely confident
before it begins implementing new technologies into its workflows.
</item 15>

<item 16>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
Together the plan is to deploy more than 1 million chargers globally.

Volta Trucks filed for bankruptcy proceedings in Sweden as difficulties with suppliers proved a hindrance to raising funds. Volta said the bankruptcy in August
of Proterra, an EV parts supplier, and the uncertainty over its own battery supplier means it needed to cut down the number of trucks it could produce. Volta,
which is based in Sweden but has operations in the U.K., also said it would file for bankruptcy in Britain.

Notable reads and other tidbits

Autonomous vehicles

Cruise, General Motors and Honda are launching a robotaxi service in Japan under a new joint venture. The service will launch with Origin vehicles in Tokyo in
2026.

Speaking of Cruise, the National Highway Traffic Safety Administration (NHTSA) has opened an investigation into the GM subsidiary’s AV system following several
incidents involving pedestrians in San Francisco.
</item 16>

<item 17>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
Within the enterprise sector, there’s a clear recognition of the value of AI. However, many lack the internal capabilities to develop AI solutions. This gap
presents a significant opportunity for startups specializing in AI to engage with enterprise clients. As the business landscape matures, proficiency in
leveraging AI is becoming a strategic imperative.

McKinsey reports that generative AI alone can add up to $4.4 trillion in value across industries through writing code, analyzing consumer trends, personalizing
customer service, improving operating efficiencies, and more. Ninety-four percent of business leaders agree AI will be critical to all businesses’ success over
the next five years, and total global spending on AI is expected to reach $154 billion by the end of this year, a 27% increase from 2022. The next three years
are also expected to see a compound annual growth rate of 27% — the annual AI spending in 2026 will be over $300 billion.
</item 17>

<item 18>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
8 predictions for AI in 2024 How will AI impact the US primary elections? What's next for OpenAI? Here are our predictions for AI in 2024.

This last year was a banger for AI as the technology went from niche to mainstream about as fast as anything ever has. 2024, however, will be the year when the
hype runs full-steam into reality as people reckon with the capabilities and limitations of AI at large. Here are a few ways we think that’s going to play out.

OpenAI becomes a product company

After the leadership shake-up in November, OpenAI is going to be a changed company — perhaps not outwardly, but the trickle-down effect of Sam Altman being more
fully in charge will be felt at every level. And one of the ways we expect that to manifest is in “ship it” mindset.

We’ll see that with the GPT store, originally planned for launch in December but understandably delayed due to the C-suite fracas.
</item 18>

<item 19>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
And for all their talk of openness, IBM and Meta aren’t exactly the poster children for the future that the Alliance’s release depicts — casting doubt on their
sincerity.

Perhaps I’m wrong and the AI Alliance will be a smash success. Or perhaps it’ll crumble under mistrust and its own bureaucracy. We’ll see; time will tell.
</item 19>

<item 20>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
The most recent one left a woman stuck under a Cruise robotaxi after being hit by a human-driven vehicle.

Foxconn and Nvidia are building “AI factories” to help accelerate AVs, robotics and other smart applications. The AI factories position the two against Tesla,
which is building the Dojo supercomputer to do more or less the same thing — take in vast amounts of data, train it, tweak code and send it back out to
self-driving cars.

Waymo released a lightweight simulator called Waymax for the AV research community. The simulator is designed to train multiple agents to perform complex,
realistic behaviors.

Electric vehicles, charging & batteries

BMW Group says it will adopt the NACS charging standard in the U.S. and Canada. Drivers of BMW, Mini and Rolls-Royce brands will gain access to Tesla
Superchargers in early 2025, and in that same year, BMW says it will implement NACS in EVs sold in the U.S.
</item 20>

<item 21>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
(Spoiler alert: Yes.)

Rumors are swilling that OpenAI may be considering developing its own AI chips. That’s going to get interesting, especially if you’ll recall that Nvidia’s
ongoing stock market rally is likely driven by the current boom in AI. The company’s ChatGPT’s mobile app hit a record $4.6 million in revenue last month, but
growth is slowing. Oh, and there’s no need to shed tears for Nvidia quite yet; Brian’s piece breaking down how Nvidia became a major player in robotics is super
interesting.

Adobe has doubled down on its Firefly generative AI models. This week, Frederic covered how the software can now generate more realistic images and can help
artists create vector graphics in Illustrator. Neato.

More on the AI front

If AI can’t go to the mountain, the mountain will come to AI: Dutch startup Tidalflow exits stealth with backing from Google’s Gradient Ventures. It is aiming
to help any software play nice with ChatGPT and other LLM ecosystems.
</item 21>

<item 22>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
I think the leadership at Hyundai — the chairman visited me last week, and we’re in close touch with him and the people who work with him — think that getting
into the 22nd century is an important thing to do and we’re getting started on it now.”

Pratt adds, “There’s actually this alignment that’s happening between cars and robotics, as well. Rod Brooks was the one who a few years ago said that modern
cars are eldercare robots. I think that’s really true. You think about the amount of computers in them, the amount of software that’s in them. TRI has different
divisions inside of it, one of which works on very advanced things in cars that is completely overlapping with some of the software and some of the concepts
that we’re using in the robotics space.”

For its part, the Boston Dynamics AI Institute lists three key pillars of its research: intelligence, dexterity and mobility. That’s effectively making robots
that are smarter, better at mobile manipulation and move more dynamically.
</item 22>

<item 23>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
It is aiming to help any software play nice with ChatGPT and other LLM ecosystems.

What big eyes you have: Adobe continues to push for easier image editing, showing off its Project Stardust as a sneak preview of its next-gen AI photo editing
engine.

Holding back the tide: Creatives across industries are strategizing in a campaign that targets potential corporate abuse of AI technology. The conglomeration is
realistic about the ways that musicians and some other creatives could benefit on an individual level from automating parts of their work. The goal is that AI
tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

Tech you can touch

A while back I argued that Apple’s new AR headset is a game-changer for startups. It seems that’s likely the case at the high end. But more in the realm of
affordability, Meta Quest 3 takes a step closer to mainstream AR/VR, Brian reports.
</item 23>

<item 24>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
Certainly most of the R&D budget inside of Toyota is spent figuring out how to make the next car or maybe the car five years from now. But there’s this notion —
and I’m sure Hyundai shares this — that we’re at the once-in-a-century time for transformation in the car industry. Battery electric vehicles are much simpler
to make than cars have been in the past — no engine, no transmission, etc. So, we’re going to have to compete in a much fiercer way in cars. But also, can we
use the skills, the dreams and the hopes of the company to go beyond cars? TRI’s job is to actually think about [what’s] next. What’s next after cars, as well
as some fancy stuff on cars.”

For TRI, much of the “what’s next” focuses on supporting aging populations. The organization invests a good portion of its resources to building out technology
designed to help older people live more independently. That’s at the root of the research we shared from the institute last week.
</item 24>

<item 25>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
But while a few egregious examples of misuse will be punished, a lack of relevant laws specific to it means that it will necessarily only haphazardly be brought
to court. On that note…

Early adopters take new rules by the horns

Big moves like the EU’s AI Act could change how the industry works, but they tend to be slow to take effect. That’s by design, so companies don’t have to adjust
to new rules overnight, but it also means that we won’t see the effect of these big laws for a good while except among those willing to make changes
preemptively and voluntarily. There will be a lot of “we are beginning the process of…” talk. (Also expect a few quiet lawsuits challenging various parts of
laws.)

To that end we can expect a newly flourishing AI compliance industry as the billions going into the technology prompt matching investments (at a smaller scale,
but still considerable) in making sure the tools and processes meet international and local standards.
</item 25>

<item 26>
Robot startups see huge market in replacing human workers: 'We can sell millions of humanoids, billions maybe' (Fortune, 2023-11-05T17:31:33+00:00)
“Forever more, human- centric robots like that are going to be part of human life. So that’s pretty exciting.”

—-

AP writer Haleluya Hadero contributed to this report.
</item 26>

<item 27>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
That decision was surprising to me based on how sources had described an all-hands meeting earlier in the week that was led by co-founder and CEO Kyle Vogt. In
that meeting, which came the day after the DMV suspended Cruise’s permit, Vogt and other leaders told staff the company had not paused operations elsewhere
besides California and gave no indication that the company was planning to. Instead, Vogt told employees the company was re-evaluating how it discloses
information to regulators to ensure it is clearly communicated, according to an account from sources who heard the call.

Cruise had even quietly launched driverless operations in Miami (just a few vehicles), a move that suggested the company was moving ahead despite its
significant problems in California.

What changed? Perhaps Cruise execs were pressured by GM or they looked around and realized that they were losing support from other states.
</item 27>

<item 28>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Either way, Cruise said it’s now going to examine “processes, systems, and tools and reflect on how we can better operate in a way that will earn public trust.”

That might be a hefty challenge, especially in California. As the Cruise drama unfolded, opposition against robotaxis grew in cities like Los Angeles. And two
of the biggest groups to oppose robotaxi expansion in California are now formally working together.

Want to reach out with a tip, comment or complaint? Email Kirsten at kirsten.korosec@techcrunch.com or Rebecca at rebecca.techcrunch@gmail.com.

Reminder that you can drop us a note at tips@techcrunch.com. If you prefer to remain anonymous, click here to contact us, which includes SecureDrop
(instructions here) and various encrypted messaging apps.

Micromobbin’

Taiwanese battery swapping giant Gogoro came to play at the Japan Mobility Show 2023, showing off how its scooter batteries can also be used to power a tiny
car.
</item 28>

<item 29>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Marketing claims made for machine learning systems that companies adopted in order to not fall behind will receive their quarterly and yearly reviews… and it’s
very likely they will be found wanting.

Expect a considerable customer withdrawal from AI tools as the benefits fail to justify the costs and risks. On the far end of this spectrum, we are likely to
see lawsuits and regulatory action with AI service providers that failed to back up their claims.

While capabilities will continue to grow and advance, 2023’s products will not all survive by a long shot, and there will be a round of consolidation as the
wobblier riders of the wave fall and are consumed.

Apple jumps in

Apple has an established pattern of waiting, watching and learning from other companies’ failures, then blowing in with a refined and polished take that puts
others to shame. The timing is right for Apple to do this in AI, not just because if it waits too long its competition may eat up the market, but because the
tech is ripe for their kind of improvement.
</item 29>

<item 30>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
2023 saw a few tentative attempts at this “agent” approach, but none really caught on. We don’t really expect any to really take off in 2024, either, but
agent-based models will show their stuff a little more convincingly than they did last year, and a few clutch use cases will show up for famously tedious
processes like submitting insurance claims.

Video and audio will also find niches where their shortcomings aren’t quite so visible. In the hands of skilled creators, a lack of photorealism isn’t a
problem, and we’ll see AI video used in fun and interesting ways. Likewise, generative music models will likely make it into a few major productions like games,
again where professional musicians can leverage the tools to create an unending soundtrack.

The limits of monolithic LLMs become clearer

So far there has been great optimism about the capabilities of large language models, which have indeed proved more capable than anyone expected, and have grown
correspondingly more so as more compute is added. But 2024 will be the year something gives.
</item 30>

<item 31>
Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation (TechCrunch, 2023-10-28T13:04:25+00:00)
This means we can list items for sale on Amazon.com more quickly, benefiting both sellers and customers. When orders are placed, Sequoia also reduces the time
it takes to process an order through a fulfillment center by up to 25%, which improves our shipping predictability and increases the number of goods we can
offer for Same-Day or Next-Day shipping. Obviously this is all a matter of reducing delivery times — also the driving factor in the company’s Prime Air
investments. The company has already set next- and same-day delivery expectations in many areas, so one wonders when we arrive at the point where any additional
time savings becomes effectively negligible. I suspect if you were to put the question to Amazon, they would say “never.”

I didn’t get to that specific question during my time with Amazon Robotics chief technologist, Tye Brady. Instead, our conversation primarily focused on three
important (I think) topics. The first is the company’s pilots with Agility’s Digit systems.
</item 31>

<item 32>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
Product design is in a moment of profound change and redefinition as technologies like artificial intelligence (AI) and spatial computing dramatically affect
computing experiences. AI, especially, may have only minor impacts on interface design but will significantly impact the holistic product or ecosystem
experience. Spatial computing, on the other hand, will change human-computer interactions and unmoor our understanding of what a computer is.

In this innovation cycle, product design requires a wider perspective of platforms and interconnections between technologies, creating a strong need for
technologists and designers to be in the process together.

For successful products and businesses, innovation is perpetual. There is a never-ending search to find the next new thing that enhances the user experience,
extends product reach, expands revenue, or all three simultaneously. Product design makes innovation less daunting and increases chances for success because it
is a multidisciplinary process with structures and frameworks to catalyze innovation. Technologists have a role in the process that expands beyond simply
validating the technology or concept, which is their customary responsibility.
</item 32>

<item 33>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
The idea, O’Hara said, is to rethink social networking from the ground up.

Ford shutters SaaS app for field work: Ford has shut down VIIZR, a software-as-a-service company that, along with Salesforce, built an app to help tradespeople
like plumbers, locksmiths and electricians to schedule field appointments, send invoices and manage customers, Kirsten reports. VIIZR, which was announced in
December 2021, was a separate company majority owned by Ford, with Salesforce as a minority investor.

Apple bets on generative AI: Apple CEO Tim Cook pushed back against the notion that the company was behind in AI on Apple’s Q4 earnings call with investors, as
he highlighted technology developments that Apple had made recently that “would not be possible without AI.” Cook also said that Apple was working on generative
AI technologies, lending credence to reports suggesting the company is on track to spend $1 billion per year on developing generative AI products.
</item 33>

<item 34>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
The new round values the Bengaluru-headquartered electric vehicle startup at about $5.4 billion

Pony.ai, the Chinese autonomous vehicle startup, scored $100 million from Neom, Saudi Arabia’s futuristic city and development project. As part of the deal, a
joint venture will be established to develop, manufacture and deploy autonomous vehicles and smart infrastructure in Neom and key markets in the Middle East
North Africa region.

Neom also announced plans this week to set up a $10 billion joint venture with Danish freight forwarder DSV.

Stellantis made a €1.5 billion deal ($1.59 billion) to take a 20% stake in Chinese electric vehicle maker Zhejiang Leapmotor Technologies, just days after
ending manufacturing in the country. The deal includes the formation of Leapmotor International, a 51% to 49% Stellantis-led joint venture that has exclusive
rights for the export and sale, as well as manufacturing, of Leapmotor products outside China.
</item 34>

<item 35>
Tesla ‘digs its own grave with the Cybertruck,’ Convoy collapses and Rivian scores a win at Rebelle (TechCrunch, 2023-10-22T19:15:29+00:00)
California-based EV startup Pebble unveiled a prototype of its flagship all-electric travel trailer. The $100,000 EV is designed to support a digital nomad
looking to get lost in the wilderness — it can live off-grid for seven days.

Commercial EV startup REE Automotive has reported an order book that now totals $25 million for its modular battery EV platforms, dedicated to B2B customers.

About 10,000 of Rivian’s all-electric vans are delivering packages throughout the U.S. for Amazon.

Tesla has urged the Biden administration to adopt stricter fuel economy standards than the NHTSA has proposed. Most other automakers have already said the
NHTSA’s proposal was unfeasible, so Tesla’s call on regulators to double down seems to be yet another way the EV maker can one-up its competitors.

Toyota has also joined the NACS bandwagon. The automaker will build certain Toyota and Lexus vehicles from 2025 onward with an NACS port.
</item 35>

<item 36>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
The Boston Dynamics AI Institute is still in its infancy, and therefore doesn’t have much in the way of public facing research to show just yet. It is, however,
doing a lot of hiring. This week it announced that MIT’s Kate Darling will lead research around “ethics and societal impact of robotics and AI.” From the
institute:

Darling’s team will explore immediate as well as long term questions on the implementation and use of robotics, impact on the workplace, infrastructure and
other topics. The team will perform studies and experiments designed to generate data needed for others to make informed ethics and policy decisions, and will
also develop a series of talks and workshops at the intersection of ethics, law, economics and robotics that will offer a platform for broad discussion.

“I’ve heard Gill say that making a car is taking small bits of metal and banging them up, welding them together and attaching them,” Raibert says. “That’s a
little on the primitive side. Now we have software, robotics, AI and all that stuff, and the car companies need to embrace that.
</item 36>

<item 37>
Meta and IBM form an AI Alliance, but to what end? (TechCrunch, 2023-12-05T13:58:12+00:00)
In addition to academic partners such as the University of Tokyo, UC Berkeley, the University of Illinois, Cornell and the aforementioned Imperial College
London and Yale, Sony, ServiceNow, the National Science Foundation, NASA, Oracle, the Cleveland Clinic and Dell have pledged their participation in some form.

MLCommons, the engineering consortium behind MLPerf, the benchmarking suite used by major chip manufacturers to evaluate their hardware’s AI performance, is
also a founding AI Alliance member. So are LangChain and LlamaIndex, two creators behind some of the more widely used tools and frameworks for building apps
powered by text-generating AI models.

But without the participation of so many major AI industry players — and lacking deadlines or even concrete objectives — can the AI Alliance succeed? What would
success look like, even?

Beats me.

The vast number of competing interests — from healthcare networks (Cleveland Clinic) to insurance providers (Roadzen) — won’t make it easy for the Alliance’s
members to coalesce around a single, united front.
</item 37>

<item 38>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Legal cases build and break

We saw a fair number of lawsuits filed in 2023, but few saw any real movement, let alone success. Most suits over copyright and other missteps in the AI
industry are still pending. 2024 will see a lot of them fall by the wayside, as companies stonewall critical information like training data and methods, making
allegations like the use of thousands of copyrighted books difficult to prove in court.

This was only the beginning, however, and many of these lawsuits were filed essentially on principle. Though they may not succeed, they may crack the process
open far enough during testimony and discovery that companies would rather settle than have certain information come to light. 2024 will bring new lawsuits as
well, ones pertaining to misuse and abuse of AI, such as wrongful termination, bias in hiring and lending, and other areas where AI is being put to work without
a lot of thought.
</item 38>

<item 39>
Cruise hits the brakes on driverless, UAW makes progress and more EV backpedaling (TechCrunch, 2023-10-30T10:15:24+00:00)
Anyone spotting a trend here?

Nikola, the electric and hydrogen-powered heavy truck maker, was awarded $165 million from its founder and former executive chairman, Trevor Milton, in an
arbitration proceeding.

Nio has opened its 2,000th Power Swap Station in China, nearing its goal to build 2,300 stations by the end of 2023. The company has expanded on its strategy of
swapping out EV batteries, rather than charging them, an infrastructure-intensive process that has the potential to make topping up a battery as quick as
filling up a gas tank.

Tesla has the attention of the U.S. Department of Justice — again. This time it has received requests for information, including subpoenas from the DOJ related
to perks, the advertised range of its EVs and personnel decisions.

In-car and mobile tech

Google Maps and Waze stopped live traffic updates in Israel and the Gaza Strip at the request of the Israel Defense Forces.
</item 39>

<item 40>
Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation (TechCrunch, 2023-10-28T13:04:25+00:00)
It’s wherever [MIT professor Julie Shah] wants to take it. We have a lot of interest in how people perceive robotics, because people will be using our robotics.
And if it is intimidating, or there’s friction there, and you don’t want to use it, then we’re failing in our design.
</item 40>

<item 41>
Robotics Q&A with Boston Dynamics’ Aaron Saunders (TechCrunch, 2023-12-09T16:05:51+00:00)
For the next few weeks, TechCrunch’s robotics newsletter Actuator will be running Q&As with some of the top minds in robotics. Subscribe here for future
updates.

Part 1: CMU’s Matthew Johnson-Roberson

Part 2: Toyota Research Institute’s Max Bajracharya and Russ Tedrake

Part 3: Meta’s Dhruv Batra

This time it’s Boston Dynamics CTO, Aaron Saunders. He has been with the company for more than 20 years, most recently serving as its vice president of
Engineering.

What role(s) will generative AI play in the future of robotics?

The current rate of change makes it hard to predict very far into the future. Foundation models represent a major shift in how the best machine learning models
are created, and we are already seeing some impressive near-term accelerations in natural language interfaces. They offer opportunities to create conversational
interfaces to our robots, improve the quality of existing computer vision functions and potentially enable new customer-facing capabilities such as visual
question answering.
</item 41>

<item 42>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
The “app store for AI” will be pushed hard as the platform to get your AI toys and tools from, and never mind Hugging Face or any open source models. They have
an excellent model to work from, Apple’s, and will follow it all the way to the bank.

Expect more moves like that from 2024’s OpenAI as the caution and academic reserve that the previous board exerted gives way to an unseemly lust for markets and
customers.

Other major companies with AI efforts will also follow this trend (for instance, expect Gemini/Bard to horn in on a ton of Google products), but I suspect it
will be more pronounced in this case.

Agents, generated video and generated music graduate from quaint to experimental

Some niche applications of AI models will grow beyond “eh” status in 2024, including agent-based models and generative multimedia.

If AI is going to help you do more than summarize or make lists of things, it’ll need access to things like your spreadsheets, ticket buying interfaces,
transportation apps and so on.
</item 42>

<item 43>
A provocative role for technologists in product innovation (TechCrunch, 2023-12-21T16:35:31+00:00)
The specifics of the process and technologies are only important as they pertain to how well they solve the problem, the costs to do so, and the overall end
user experience.

Product innovation is experimental and should not always be expected to yield productive results. It requires a learning curve and patience, as the outcomes are
often ambiguous and unknown. Business leadership can struggle with this perspective because it is indefinite (in terms of results and timelines), and it is
challenging to translate pure technology innovation into value creation. A gap opens between technology and product teams wherein technology teams strain to
articulate the capabilities and value of a technology innovation, leading to unfilled promises and the perception of “technology for the sake of technology” or
quips like “a solution looking for a problem.”

The current hype cycle in AI serves as a great concrete example. For technology or product executives, the challenge is how to do more than check the AI box —
it’s about how to meaningfully incorporate AI into a product. Instead of forcing a technology onto a product, the design process flows to the technologies.
</item 43>

<item 44>
Robotics Q&A with Boston Dynamics’ Aaron Saunders (TechCrunch, 2023-12-09T16:05:51+00:00)
When would you pay as much for a robot as you would a car? When it achieves the same level of dependability and value you have come to take for granted in the
amazing machines we use to transport us around the world.

What important robotics story/trend isn’t getting enough coverage?

There is a lot of enthusiasm around AI and its potential to change all industries, including robotics. Although it has a clear role and may unlock domains that
have been relatively static for decades, there is a lot more to a good robotic product than 1’s and 0’s. For AI to achieve the physical embodiment we need to
interact with the world around us, we need to track progress in key technologies like computers, perception sensors, power sources and all the other bits that
make up a full robotic system. The recent pivot in automotive towards electrification and Advanced Driver Assistance Systems (ADAS) is quickly transforming a
massive supply chain. Progress in graphics cards, computers and increasingly sophisticated AI-enabled consumer electronics continues to drive value into
adjacent supply chains.
</item 44>

<item 45>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
Three years later, Pratt was named TRI’s CEO. Raibert continued as Boston Dynamics’ CEO until 2019, handing the reins over to longtime employee Rob Playter, who
managed the company’s shift to more commercial pursuits. Last year, Raibert (who still serves as BD’s chairman) founded the Boston Dynamics AI Institute. The
organization has a lot in common with TRI. Both are dedicated to pure research, with the backing of two major automakers (Toyota and Hyundai).

I recognize that “pure research” is a bit of a loaded term. The concept is certainly a hopeful one: finding a method that can sustain research outside of the
pressures of academia and corporate R&D.

Asked whether there is pressure from Toyota to productize, Pratt says, “There actually is not, and I’m not saying that just to be nice. I think one of the great
things about companies that have existed for long times is that this isn’t the first time that they’ve done R&D.
</item 45>

<item 46>
SoftBank Corp takes 51% of Cubic Telecom for $513M to drive into the connected car world (TechCrunch, 2023-12-05T07:24:23+00:00)
But the size of this particular round is due to how fast the company is growing right now. Cubic Telecom is currently linking up 450,000 new vehicles — consumer
cars, trucks and more — using its platform each month, and orders in the books are going to raise that rate “exponentially” over the next five years, the
company said. Cubic Telecom cites forecasts from McKinsey noting that some 95% of new vehicles will be sold with wireless connectivity built into them by 2030.
Juniper puts the number of new cars on the road by then that will be “connected” at just under 400 million.

Growth in the industry is due to a few factors: networks continue to improve, in particular with 5G especially suited to IoT deployments; cloud-based data
services and vehicles themselves are getting more advanced; and thanks to the evolution of other connected devices like smartphones, watches, televisions and so
much more, consumers and business are expecting more functionality in their vehicles.
</item 46>

<item 47>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
And that’s all some people need.

Likely there will be some half-hearted efforts to block generated content from being used in this way, but these posts can’t be taken down fast enough by the
likes of Meta and Google, and the idea that X can (or will) effectively monitor and take down such content is implausible. It’s gonna be a bad time!
</item 47>

<item 48>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
But 2024 will be the year something gives. Where exactly it is impossible to predict, as research is active at the frontiers of this field.

The seemingly magical “emergent” capabilities of LLMs will be better studied and understood in 2024, and things like their inability to multiply large numbers
will make more sense.

In parallel, we will begin to see diminishing returns on parameter counts, to the point where training a 500-billion-parameter model may technically produce
better results, but the compute required to do so could provably be deployed more effectively. A single monolithic model is unwieldy and expensive, while a
mixture of experts — a collection of smaller, more specific models and likely multimodal ones — may prove almost as effective while being much easier to update
piecemeal.

Marketing meets reality

The simple fact is that the hype built up in 2023 is going to be very hard for companies to follow through on.
</item 48>

<item 49>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
Unfortunately for anyone hoping for substantive federal regulation in the U.S., 2024 is not the year to expect movement on that front. Though it will be a year
for AI and everyone will be asking for new laws, the U.S. government and electorate will be too busy with the trash fire that will be the 2024 election.

The 2024 election is a trash fire and AI makes it worse

How the 2024 presidential election will play out is, really, anyone’s guess right now. Too many things are up in the air to make any real predictions except
that, as before, the influence mongers will use every tool in the box to move the needle, including AI in whatever form is convenient.

For instance, expect bot accounts and fake blogs to spout generated nonsense 24/7. A few people working full time with a text and image generator can cover a
lot of ground, generating hundreds of social media and blog posts with totally fabricated images and news.
</item 49>

<item 50>
Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation (TechCrunch, 2023-10-28T13:04:25+00:00)
So the jobs changed. The jobs exist, but it’s a big sweeping change.

If I visit the labs, I’ll see these sorts of experiments in action.

Yeah. If you were to go to outside of Nashville today, you would see Proteus working with our Cardinal arm. You’d get to see the interoperability. We have the
Proteus drives moving carts to the outbound docks. If you were to go down to Hou 6 just outside of Houston, you would see Sequoia fulfilling orders today, right
in time for holiday shopping.

What role do people play in that picture?

People will always be at the center of a robotics universe. We know more robots, more jobs that we see through the productivity increases that we have.

[The MIT study] sounds like it’s largely about perception and what people think of robots, rather than job numbers specifically.

I’m not sure. It’s wherever [MIT professor Julie Shah] wants to take it.
</item 50>
=== END ITEM id=67051dcb55929978 ===
