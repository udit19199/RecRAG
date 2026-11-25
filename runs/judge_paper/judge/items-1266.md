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

This file has 7 items.

=== ITEM id=103fa34e289d4555 ===
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
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
The over-ear headphones may not offer all the bells and whistles you’d get with a pair of Sony’s premium noise-canceling headphones, but they offer decent ANC
and up to 35 hours of battery life.

The best tablet and laptop deals

2021 iPad (64GB, Wi-Fi) $ 249 $ 329 24 % off $ 249 $ 249 $ 329 24 % off Apple’s entry-level iPad from 2021 has a 10.2-inch screen, an A13 Bionic chip, and a
12MP front camera that supports Apple’s Center Stage feature. It’s also compatible with the first-gen Apple Pencil and the Smart Keyboard and is the only iPad
that still has a 3.5mm headphone jack. $249 at Amazon$249 at Walmart

The newer 10th-gen iPad is also on sale right now for $399.99 ($50 off) with 64GB of storage and Wi-Fi at Best Buy and Walmart.
</item 1>

<item 2>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
It’s certainly a system that’s likely to last you a while. It’s true that Apple is currently on a one-year upgrade cycle for it first-party M-series chips, but
it’s genuinely tough to imagine and advanced as pronounced as the one it made from Intel to in-house silicon happening any time soon. There’s no such thing as a
future-proofed laptop, so the best you can hope for is one that will give you good mileage. That certainly applies here.

Along with building new chips, Apple has spent the last few years listening to consumers in a way it hadn’t for decades. That means finally updating the camera,
building a better keyboard (the class action settlement probably tipped the scales as well) and ditching the well-meaning but ultimately ineffectual Touch Bar
altogether.
</item 2>

<item 3>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
Discounts on Apple products are few and far between. Luckily, with Amazon's second big sale event of the year (officially called Prime Big Deal Days), we've
found a variety of Prime Day Apple deals. That includes iPads, Apple Watches, iPhone cases, and MagSafe accessories. These deals aren't just at Amazon
either—we've linked to other retailers like Best Buy when available.

WIRED Featured Deals

Updated October 11: We added new deals on the Apple Magic Keyboard, Apple Studio Display, ESR HaloLock 3-in-1 Travel Wireless Charging Set, Monoprice Extra Wide
Microfiber Desk Mat, and Apple AirTags.

Table of Contents

We test products year-round and independently recommend the best. If you buy something using links in our stories, we may earn a commission. This helps support
our journalism. Learn more.

iPad Deals

If you'd like more info, check out our Best iPads guide for an overview of every iPad model.
</item 3>

<item 4>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It's dipped lower to $899 in the past, and it hits this price often, but we still think this is a solid deal. However, if you want the latest version, it's
worth noting Apple is rumored to be launching an M3-powered version of the 13-inch and 15-inch MacBook Air next year. Keep in mind that this model drops to this
sale price relatively often.

If you want a bigger screen, the 15-inch MacBook Air (8/10, WIRED Recommends) is the first large-screen option within the Air lineup. It packs all the same
features as the 13-inch version including a redesigned chassis and 1080p webcam. The M2 also delivers excellent performance and solid battery life (with about
10 hours of use). For the smoothest experience, we recommend opting for 16 gigabytes of RAM if it's in your budget. It's also on sale for $1,449 ($250 off).
</item 4>

<item 5>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It doesn't pack as many ports as the version with the M2 Pro, but you'll still get two Thunderbolt 4 USB-C ports, two USB-A, an HDMI, a 10-gigabit Ethernet
connection, and a 3.5-mm headphone jack. It's dipped this low a few times before, but it's still a great deal.

This is our favorite external keyboard in our Best MacBook Accessories guide, especially if you're a fan of the Magic Keyboard on your MacBook and are looking
to fully replicate your setup. This one comes with a Touch ID button, which you can use to log in, authenticate for purchases, and download apps. It also has
the same function row keys as the M-series MacBooks, including playback buttons, a Mission Control Key, and a dedicated Do Not Disturb button.

With the ability to flex the stand up to 22 inches high, you can adjust this laptop stand depending on your setup—whether you’re using an external monitor or
working solely off a MacBook.
</item 5>

<item 6>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
It has only been a year since the most recent iPad Pro debuted. Other than a few small tweaks, Apple only really swapped in the M2 in place of the M1. The next
iPad Pro will probably need more changes to make it more palatable for anyone who picked up the previous two models, which means Apple might need more time to
get it ready. If that's the case, the company might be content to give the iPad Pro a spec bump for the time being.

Other iPads could also get a refresh at the event, just in time for the holidays. We won't have to wait too long to find out either way. We'll have full
coverage of everything you need to know from the Scary Fast event at Engadget, so while you're waiting for Halloween trick-or-treaters to show up, slip into
your jammies, grab a mug of cocoa and join us on Monday evening.
</item 6>

<item 7>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
This is also the lowest price we've tracked, yet.

If you're constantly working on the go, a portable monitor is a great way to get more screen real estate. This one from Mobile Pixels attaches to your laptop
screen using magnets (you'll have to be comfortable with sticking and leaving adhesive magnets on the back of your laptop) and connects via a USB-C cable.
Simply slide it out to reveal a 13-inch display. It adds some bulk and thickness to your laptop, but you can remove it when it's not in use.

We love this laptop stand for those who prefer working from your bed or couch. The stand's height isn't adjustable, but you have the option to raise the base to
position it at the best angle for typing, watching a movie on a tablet, or sketching. There's also extra space for your coffee mug and a small drawer for
storing knickknacks.

Twelve South StayGo Mini USB-C Hub Photograph: Twelve South

We've used this hub with iPads, but it's also great for MacBooks.
</item 7>

<item 8>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The Ultra, for example, is for all in intents and purposes, two Maxes stitched together. There are other factors at play as well, like the fact that Mac Studio
and Mac Pro do a better job with heat dissipation, while the MacBook Pros are more equipped than the fan-less Airs, which throttle performance more quickly due
to heat.

There are still a handful of ways, however, that even the baseline M3 will outperform the M2 Ultra. Two clear instances are ray tracing and mesh shading, which
were previously software-accelerated.

With Blackmagic Disk Speed, which measures SSD performance, the 16-inch Pro scored 6367 write and 5057 read speeds, vs. the Mac Studio’s 6648/5716. Running
Geekbench 6, the system scored 3149 on the single-core test and 21269 multi-core. That’s a nice bump over the M3 iMac’s 3024/11810.
</item 8>

<item 9>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
It’s more than capable for both work and play and should give you years of dependable use. But, again, the M3-enabled Airs are likely on their way in 2024. So
if you know you’re drawn to the newest thing, you might want to wait.

MacBook Air (13-inch, 2020, M1)

The MacBook Air with the M1 chip is a couple years old at this point, but still officially part of Apple’s laptop lineup. It dropped to an all-time low of $750
for October’s Prime Day at Amazon and now Amazon and Best Buy are selling it for the same price. It’s the budget pick in our guide to MacBooks and the top pick
in our guide to budget laptops. Check out our review from when it first came out in which we praised its quick speeds, excellent screen and good battery life.
</item 9>

<item 10>
The 26 best Amazon Prime Day deals you can still shop for today (Engadget, 2023-10-12T15:44:16+00:00)
Other SKUs with more RAM and storage are also on sale. $1,049 at Amazon

The 15.3-inch version of Apple’s MacBook Air is still down to $1,049 for an entry-level model with 8GB of RAM and a 256GB SSD. That matches the best price we’ve
seen and takes $250 off Apple’s list price. The MacBook Air is the top pick in our guide to the best laptops, and we gave the 15-inch model a review score of 96
earlier this year, as it covers just about all of the things we look for in a quality mainstream notebook. Compared to the 13.6-inch version, it also includes a
more powerful speaker system and a slightly upgraded GPU. The SSD in this base configuration is technically slower than the one in higher-capacity SKUs, but
that shouldn’t be a major problem unless you’re looking to do more complex work.
</item 10>

<item 11>
The best Xbox controller to buy right now (The Verge, 2023-11-21T20:04:33+00:00)
You can get all of those in a $59.99 accessories pack, which turns the Elite Series 2 Core back into a standard Elite Series 2, but that sets you back $10 more
than just going for the Elite Series 2 to start with.

You must also be aware that the Elite has a bit of a reputation for lackluster quality control, with horror stories from users about going through multiple
replacements under warranty. Microsoft extended the controller’s warranty from 90 days to a year in late 2020 to help address concerns, but buying an Elite
Series 2 may still feel like a slightly risky proposition. However, when you nail that perfect sequence — switching to your pistol and landing a headshot with a
quick pop-pop of the hair trigger while your thumbs stay firmly planted on the sticks — it can feel worth it.
</item 11>

<item 12>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The range still tops out at $649, but you’re getting a lot more for your money: All of the hardware improvements, 1TB of storage instead of 512GB and an
improved carry case that can pop out into a slimmer shell that’s smaller and easier to fit in a bag. As mentioned, it also has the same anti-glare etched glass
as the outgoing 512GB LCD model.

Finally, there’s also a Limited Edition version of the 1TB model, with a translucent shell and a customized carry case, which will be sold at $679. We imagine
stocks will not last very long, as it looks pretty rad:

Valve

While there are no concrete updates on future hardware developments, Valve engineer Pierre-Loup Griffais noted that, so long as interest in handheld gaming
remains high, the company has "plans for successive generations of handhelds." For now, Griffais explained, it's important to Valve that developers don't need
to "split their attention" between different performance targets.
</item 12>

<item 13>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
Apple has at least one more trick — or possibly a treat — up its sleeve this year. On October 30, the night before Halloween, the company is hosting another
event. This one is dubbed “Scary Fast,” which is a strong indicator that Apple will have some speedy new hardware to show off. Unusually, Apple is holding this
event in the evening. It starts at 8PM ET.

This year's iPhones and Apple Watches are out in the wild already, but Apple has a bunch of other product categories that it could focus on here. There's always
the chance that we might get iPad updates. But, if rumors are to be believed, Apple's focus during the Scary Fast event will be on Macs and a fresh batch of
chipsets to power them.

M3 chip

Apple won't be talking about speed advancements without having new hardware to discuss. That could all start with upgraded chipsets.
</item 13>

<item 14>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
We're past the point of the year when Microsoft would normally announce a refresh, but the current model should still be useful if you're coming from an older
Surface Pro that's in desperate need of an upgrade.

Lenovo IdeaPad Flex 5i Chromebook Plus

Lenovo's IdeaPad Flex 5i Chromebook Plus is on sale for $379 at Best Buy, which is $120 off its list price. This is the most recent version of the top pick in
our Chromebook buying guide: Its 14-inch, 1,920 x 1,200 IPS touchscreen is good for the money, it's plenty fast for the kind of web browsing and light work
you'd do with Chrome OS and both its keyboard and trackpad are comfortable. This model comes with a Core i3-1315U processor, 8GB of RAM, 128GB of eMMC storage,
one USB-A port, two USB-C 3.2 ports and a microSD card slot.
</item 14>

<item 15>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
There's also a 13-inch/14-inch version for those who have smaller MacBooks or want a sleeve for their iPads.

Speck’s Smartshell case has the proper cutouts for ports and rubberized feet to keep your machine in place. You can throw it in your backpack, tote, or duffle
bag without worrying about damaging your machine. We tested it with the 14-inch MacBook Pro but it’s also available for a wide range of MacBook models. It’s
worth noting, that the hardshell isn’t scratch-resistant (despite Speck’s claims)—we’ve noticed a few scrapes on ours here and there, but it’s not too
noticeable.

This keyboard is a runner-up in our Best MacBook Accessories guide. It mimics the MacBook keyboard with a Do Not Disturb key, and the ability to program the
function row key with Apple's own apps like Final Cut Pro, Photos, Safari, and more, using the Logi Options+ app—in addition to other Mac-specific shortcuts.
</item 15>

<item 16>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
It packs a 10.9-inch LCD screen, a USB-C port, slim bezels, Touch ID built into the top power button, and the same M1 processor as the 2020 MacBook Air. It also
has support for both the second-generation Apple Pencil, Smart Keyboard Cover, and Magic Keyboard Cover. It dipped lower to $460 in July, but we still think
this is a solid deal.

iPad Pro Photograph: Apple

Apple's iPad Pro tablet (7/10, WIRED Review) packs incremental upgrades compared to its predecessor, but it's the best option if you want the ultimate iPad. The
large, 12.9-inch LCD screen comes equipped with ProMotion (which offers a 120-Hz screen refresh rate for smoother scrolling), Face ID, four speakers for better
sound quality, and more microphones to pick up your voice more clearly.
</item 16>

<item 17>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
Its M2 processor didn’t kick-start a revolution like the M1 generation, but it’s a great performer for any user outside of more demanding creatives.

It does have some slight downsides, like slower storage in the base 256GB configuration and a notch cutout in its otherwise excellent screen. But even so, there
hasn’t been a more travel-friendly laptop offered by Apple since the days of the polarizing 12-inch MacBook, and this one’s good enough to be the No. 1 laptop
we now recommend in our buying guide.

The newer 15-inch MacBook Air with M2 is a lot like a blown-up 13-inch Air, and judging from our glowing review, there’s nothing wrong with that. Like its
smaller counterpart, it has an M2 chip, two Thunderbolt 4 / USB-C ports, MagSafe charging, and its base model comes with a 256GB SSD and 8GB of RAM.
</item 17>

<item 18>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
Back in April 2021, the company finally started bumping its webcams from 720p to 1080p. It was a long-awaited refresh for a company that puts so much emphasis
on its mobile imaging. The new hardware was coupled with improvements brought on by the M1’s on-board digital signal processing. While the camera hardware has
remained the same since, updates to the chips have further tweaked picture quality. As it stands now, the webcam is totally fine, and the addition of portrait
and studio lighting modes further improve its. Portrait mode does still have some trouble with edges. I wear over-ear headphones while podcasting, and it hasn’t
quit figured out what to do with the gap between the band and my head.

Frankly, however, Apple’s biggest enemy here is itself. Continuity Camera is a game changer for many. The system lets a mounted iPhone take over webcam duties,
and there’s really no comparing the two. I never go on a trip without the Belkin MagSafe camera mount.
</item 18>

<item 19>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
Looking for a new Apple device? There's no time better than now. Whether you're on the hunt for a new iPad, MacBook, Apple Watch, or just a new iPhone case,
there are more Apple products on sale for Cyber Monday than you'll typically find at any other time of year. Here are some of our favorites. For more gadgets on
sale, be sure to check out our Best Cyber Monday Deals roundup.

We test products year-round and handpicked these deals. The discounts we show are based on actual street prices at retailers in the past few months. Products
that are sold out or no longer discounted as of publishing will be crossed out. We'll update this guide periodically.

Updated November 27: We've added a new deal on a 14-inch MacBook Pro (M3 Pro, 2023) and updated pricing and live deals as Cyber Monday comes to an end.

If you buy something using links in our stories, we may earn a commission. This helps support our journalism. Learn more.
</item 19>

<item 20>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
That’s a nice bump over the M3 iMac’s 3024/11810. In our Mac Studio review, the M2 Ultra got 2819 single and 21507 multi-core. Taking a step back in time to
Geekbench 5 (which we’ve been running on M-series systems from the outset), the M3 Max gets a single-core score of 2323 and 23190 on the multi-core. That’s a
big performance increase over the M3 iMac’s 2191/10619 and the M2 Max 14-inch Pro‘s 1952/15249.

The M3 Max powering our Pro sports 16 CPU cores — that’s 12 performance and 4 efficiency — while the GPU packs in a whopping 40 cores. On Cinebench 2024, it
scored 142 for single-core and 1693 with the multi-core, compared to the M3 Max’s 138/617. You can see a pattern emerging.
</item 20>

<item 21>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The supply chain ultimately thought different(ly). What announcing the M3, M3 Pro and M3 Max at once affords the company, however, is options — though less so
for the M3-only iMac.

Between the 14- and 16-inch MacBook Pros, you’ve got all three configurations covered. Presumably an M3 Ultra is also on the way in the not-too-distant future,
but will almost certainly be confined to desktops like the Mac Studio, and Mac Pro, owing to the limitations of system architectures. Apple also recently shot
down rumors surrounding a 27-inch iMac. The 14-inch MacBook Pro can be configured with any of the three new chips, while the 16-inch is limited to just the Pro
and Max.

Apple sent us the 16-inch with the M3 Max. The company, understandably, likes to put its best foot forward with this stuff (so yes, it’s also the Space Black
model).
</item 21>

<item 22>
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
Unlike the last-gen model, the 2022 iPad sports a larger 10.9-inch screen, USB-C support, and a faster processor. Read our review.

The best ebook reader deals

The 2019 Kindle Oasis with 8GB of storage and ads is available for $199.99 ($50 off) at Best Buy. The 7-inch e-reader sports physical page-turning buttons and a
waterproof 300ppi E Ink display but lacks the USB-C support found in newer Kindles. Read our review.

The best smart speaker deals

The best smart display deals

Echo Show 5 (third-gen) $ 50 $ 90 44 % off $ 50 $ 50 $ 90 44 % off The new Echo Show 5 is just as small as its predecessor but features updated mics and a
speaker system that delivers double the bass and clearer sound quality.
</item 22>

<item 23>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
However, this version is fairly bare-bones—it doesn't come with a Touch ID button or the same function row keys as the M-series MacBooks (that model isn't on
sale, sadly). But it does come with the standard keys like playback controls, a Mission Control key, as well as a number pad. This is also the lowest price
we've tracked for this keyboard, so far.

Apple Studio Display Photograph: Apple

The Studio Display (9/10, WIRED Recommends) has a spacious 27-inch display that's perfect for juggling multiple apps at the same time and a 5K resolution that's
stunningly sharp (there's no HDR though, so colors look slightly more contrasty and saturated compared to the neutral tones on Apple's Pro Display XDR).
Although it's pricey, this is an excellent monitor for those who want a really accurate and sharp screen. Equipped with a 12-megapixel camera, along with
built-in mics and speakers, it's ideal for video calls too.
</item 23>

<item 24>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
Gurman suggests we'll see USB-C versions of the Magic Trackpad, Magic Mouse and Magic Keyboard at the event.

This makes a lot of sense. Apple ditched the Lightning port in all iPhone 15 models, while it started adopting USB-C on iPads several years back. The company
also recently unveiled an Apple Pencil that has a USB-C charging port. This change was prompted by European Union rules that require a standard charging port on
certain devices. Having the same port on all Apple devices and accessories will mean consumers have fewer types of cables to worry about.

iPad Pro

Nathan Ingraham / Engadget

While Gurman may have poured cold water on the idea, we can't disregard our hunch that we'll see at least one iPad at Apple's event. With the focus likely to be
on M3-powered products, the chances are that any new iPad will be of the Pro variety with one of those chips inside.
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
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
Apple sells MacBooks equipped with its own M-series chips in a wide range of sizes and price points. The offerings start with the 13-inch MacBook Air from 2020
at $999 and go all the way up to the latest 16-inch MacBook Pro starting at $2,499. But finding a deal on a current Mac with an M1, M2, or even the new M3 chip
— as well as the higher-end M3 Pro and M3 Max — is actually not that difficult.

While Macs may not experience perpetual discounts, it’s not uncommon to see various current models discounted by as much as $400. Alternatively, purchasing
refurbished options directly from Apple is another way to save money without as much waiting for the changing winds of deals to blow your way. Apple’s
refurbished store provides a one-year warranty on all products and generally offers discounts of up to 15 to 20 percent off the price of a new unit.
</item 26>

<item 27>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
M2 Pro MacBook Pro

As for the 14- and 16-inch MacBook Pros from early 2023 with M2 Pro / Max processors, they’re still more than capable — they may even be a better buy than the
new models if you can find them at a steep discount.

For example, the base 14-inch MacBook Pro with an M2 Pro chip, a 512GB SSD, and 16GB of RAM is selling for $1,629 ($370 off) at Amazon, which is a new low price
that may not last very long. However, the 14-inch model with 1TB of storage, a 12-core M2 Pro CPU, and 19 GPU cores is down to $1,999 (a steep $500 off) at B&H
Photo. Now we’re talking!

Sadly, the situation isn’t quite as good for the 16-inch MacBook Pro with the M2 Pro.
</item 27>

<item 28>
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
</item 28>

<item 29>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Backbone says the USB-C models will work with Apple’s new iPhone 15 lineup as well.

Crucial MX500

The Crucial MX500 internal drive is on sale for $46 right now. It's been an Engadget favorite for a while thanks to its sequential read speeds of 560MB/s and a
standard 2.5-inch design that should make it easy to swap in for your desktop's or laptop's used-up drive.

Your October Prime Day Shopping Guide: See all of our Prime Day coverage. Shop the best Prime Day deals on Yahoo Life. Follow Engadget for Prime Day tech deals.
Learn about Prime Day trends on In The Know. Hear from Autoblog’s experts on the best Fall Prime Day deals for your car, garage, and home, and find Prime Day
sales to shop on AOL, handpicked just for you.
</item 29>

<item 30>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
It does have the same M2 chip as the 13-inch MacBook Air (our top pick in our Best MacBooks guide), but it still comes with the same 13.3-inch display, 720p
webcam, and two Thunderbolt USB-C ports. Performance is great, and there's reliable battery life, but we don't suggest it for processor-intensive workloads.

MacBook Air Photograph: Apple

The 15-inch MacBook Air (8/10, WIRED Recommends) is the first large-screen option within the Air family. It packs all the same features as the 13-inch MacBook
Air including a redesigned chassis, 1080p webcam, and M2 chip—while still maintaining a lightweight and thin design. It's an excellent option for those who
spend all day working off a laptop and don't want to plug into an external monitor.

This is our favorite external keyboard in our guide to Best MacBook Accessories. It's a great option if you're a fan of the Magic Keyboard and are looking to
fully replicate your MacBook setup.
</item 30>

<item 31>
The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more (Engadget, 2023-11-27T10:00:30+00:00)
They carry a list price of $169 but often sell for closer to $150, so this is $10 off the usual. These non-Pro AirPods came out in 2021 and got a review score
of 88 from us. They pair quickly with iPhones and are a good complement to Siri’s helpfulness. They don’t have ANC however — you’ll need to go for the AirPods
Pro for that feature — but they do have Adaptive EQ which monitors and adapts sound in real time. The fit is comfortable and they’re water-resistant so they can
handle workouts or the weather.

Apple Watch Cyber Monday deals

Apple Watch Ultra 2

The Apple Watch Ultra 2 is now $739, which is $60 off. It sold for $9 less during Black Friday and is one of the few Apple products to not maintain its pricing
for Cyber Monday. Both Amazon and Best Buy are selling it for $739.
</item 31>

<item 32>
The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more (Engadget, 2023-11-27T10:00:30+00:00)
That's not a huge discount, but the streaming box rarely goes on sale. We gave it a 94 in our review because it’s fast, has an excellent Siri remote and
integrates seamlessly with Apple’s overall ecosystem. Navigation is clean and easy to use, plus the device can stream in 4K, HDR and Dolby Vision.

Mac Mini M2

The base model of Apple diminutive desktop computer, the Mac mini, is currently on sale for $100 off for Cyber Monday at Amazon, B&H Photo, Best Buy. It earned
an 86 in our review because it packs a lot of processing power, using the same M2 chip as the two current-model MacBook Airs — but it’s much cheaper than either
laptop. If you need a little more storage, you can get it with 512GB of SSD for $699 at B&H Photo.

Your Cyber Monday Shopping Guide: See all of Yahoo’s Cyber Monday coverage, here.
</item 32>

<item 33>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
If it's within your budget, we recommend the model with 16GB of RAM, which is $150 off at B&H photo. And if you upgrade to a model with the M3 Pro chip for an
even speedier experience — you'll get a $200 discount as it's down to $2,199 at B&H Photo. We gave the new M3 MacBook Pros an appreciative review and named them
the best option for creatives in our buyer’s guide.

Read our review of the new MacBook Pros with the M3 chip

MacBook Air (15-inch, 2023, M2)

The 15-inch MacBook Air arrived in June and added a little more screen real estate to an already great machine. It’s down to $1,049 for Black Friday at Amazon
and Best Buy. That’s the same as the all-time low it’s hit a few times these past couple of months. It's $200 off at B&H Photo.
</item 33>

<item 34>
Here are the best deals on headphones and wireless earbuds we could find (The Verge, 2023-12-07T22:32:23+00:00)
Plus, when paired with Apple’s Vision Pro headset, they support lossless audio.

During Black Friday, we saw steep discounts on both versions of the latest AirPods Pro, with retailers taking up to $60 off. However, these deals have since
disappeared, and you can only buy the AirPods Pro without USB-C for $189.99 in-store from the Micro Center. The second best discount is available at B&H Photo,
where anybody can order the 2022 buds without USB-C for $229 ($20 off). The buds with USB-C support, meanwhile, are only $3 off at $237 from Amazon, Walmart,
and B&H Photo— though Best Buy’s Plus and Total members can save $12.50.

AirPods (second-gen) deals

Alternatively, you can buy a pair of the second-gen AirPods with a wired charging case.
</item 34>

<item 35>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
It’s simple math. People can’t get enough of the things. The Pro has three Thunderbolt 4/USB-C ports to the Air’s two. Also, for some reason, all of the ports
(save for the headphone jack) are all on the same side with the Air. Things can get crowded fast. Both systems have the latest MagSafe connector — though I’d
like to point out that despite being a bespoke charger, you’re able to top the laptops up faster via Thunderbolt. The other important addition on the Pro side
is an HDMI port, which is key for the aforementioned external display support.

From an industrial design perspective, not much has changed here from the 2021 model. Apple settled into a look it likes, balancing rounded edges with a flat,
almost boxy design that isn’t dissimilar from the iPhone in that respect. On top is the familiar reflective Apple logo, while “MacBook Pro” is etched into the
bottom.
</item 35>

<item 36>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
I’ve begun thinking of the 16-inch MacBook Pro as something more along the lines of a portable desktop. That is to say that the M3 Max is plenty powerful for
even most professional uses. The chip also affords the laptop the ability to power up to four external displays, making it a fine workstation option for many.
When it’s time to take your work home (assuming you don’t live in France), you can pop it in your bag and take it with you. I would recommend against taking
your work (and large laptop) home with you every night, but you do you.

https://techcrunch.com/2023/11/06/apple-imac-m3-review/

The new MacBook Pro, which goes on sale this week, was announced at last week’s Scary Fast event, alongside a new iMac and — of course — several members of the
M3 line. That latter bit marked a big departure for the company, following rumors that Apple had planned the initial M3 launch for WWDC.
</item 36>

<item 37>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
GPU updates were the centerpiece of the news for several reasons. One is a feature the company calls Dynamic Caching. Apple writes,

With Dynamic Caching, only the exact amount of memory needed is used for each task. This is an industry first, transparent to developers, and the cornerstone of
the new GPU architecture. It dramatically increases the average utilization of the GPU, which significantly increases performance for the most demanding pro
apps and games.

More simply put, the idea is allocating just the right amount of local memory to a given task. It’s something Apple has been working on for a while. The other
two additions should be more familiar for anyone who’s spent time behind a game controller. The first is ray tracing, a feature designed to bring added realism
to the gaming experience, though more accurate depictions of lighting and reflections.

That feature is now hardware-accelerated, along with mesh shading. The triangle computer shader makes processing complex geometrical images more efficient.
</item 37>

<item 38>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
This helps if you’re working on pro-level tasks like video editing but can’t spend the premium that Apple charges for its bigger Pro models.

The Razer Blade 14 earned an 8/10, WIRED Recommends award in our review. It has a gorgeous 16:10 display, plenty of USB ports, a full-size HDMI output, and an
Nvidia RTX 3080 Ti, with the option to upgrade to the 40-series, giving it plenty of power to tear through the toughest games. It's the laptop WIRED reviewer
Eric Ravenscraft used through Starfield's launch weekend; it has proven its worth already.

Photograph: Das Keyboard

You may have to click on “See More Buying Options” to see this deal. The Das Keyboard MacTigr pairs well with Apple's laptops and PCs. It has a dedicated Mac
layout, Cherry MX Red switches, a two-port USB-C hub, and a high-quality, all-metal build. It rarely goes on sale.
</item 38>

<item 39>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
Have you tried typing in gloves? Not easy.

Apple describes the new color in the most Apple way possible, “The finish features a breakthrough chemistry that forms an anodization seal to greatly reduce
fingerprints.” The company hasn’t perfected the wholly fingerprint-proof surface, but the new finish does an excellent job keeping things to a minimum.

Camera

Both new Pro models are built around Apple’s Liquid Retina XDR display. The 14-inch sports 3024 x 1964 pixels to the 16’s 3456 x 2234 pixels — both come out to
a pixel density of 254 ppi. The 15-inch Air, meanwhile, has a 224 ppi density. The 16-inch system’s display is largely unchanged from the 2021 model. It has a
1000 nit sustained brightness and 1600 nit peak with HDR (high dynamic range) content. For SDR (standard dynamic range) content, however, brightness has been
bumped up from a 500 to 600 nits max.
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
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
Still, if you're set on a Fire tablet and you want a big display, this isn't a bad deal.

Best Prime Day Laptop and Accessory Deals

MacBook Air Photograph: Apple

The 2023 MacBook Air (8/10, WIRED Recommends) is one of our favorite laptops this year. It has an excellent six-speaker sound system, a 1080p front-facing
webcam, and a 10-core variant of the M2 processor the previous year's model came with. It's blazingly fast for most normal work and can even handle some light
video editing and other heavier tasks.

This is last year's 13-inch MacBook Pro (7/10, WIRED Review) with the M2 chip and Touch Bar. It has the same processor that's in the new MacBook Air (our top
pick for most people) and doesn't offer any major hardware upgrades except for a fan, which allows the processor to get a little warmer and eke out more power
over a longer period of time.
</item 41>

<item 42>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
You can, of course, spec them up the wazoo if you’re willing to pay more, as exhibited by the review unit Apple sent us of the 16-inch model, which cost an
eye-watering $7,199.

While these models are still quite new, versions that actual humans buy have begun receiving discounts. The 14-inch MacBook Pro in its base configuration with
the M3 Pro (11-core CPU / 14-core GPU), 18GB of RAM, and 512GB SSD is currently on sale for $1,749 ($250 off) at B&H Photo. A similar version with 1TB of
storage and a 12-core / 18-core M3 Pro is down to $2,149 (also $250 off) at B&H Photo. As for the base 16-inch MacBook Pro with the M3 Pro (12-core CPU /
18-core GPU), 18GB of RAM, and a 512GB SSD, it’s selling for $2,249 ($250 off) at Amazon and B&H Photo.
</item 42>

<item 43>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

There are still tons of Apple Black Friday deals you can shop for online; these are our favorites.

Most of Apple’s current-generation devices went on sale for Black Friday — and most of those deals look like they’ll carry over through the weekend and into
Cyber Monday. That means if you were otherwise occupied on Friday, you can still save on iPads, Apple Watches, MacBooks, AirPods and more. Some of the prices
are new lows and many others match the record prices we’ve seen previously. We’ve tested and reviewed nearly every device below, so check out the links to our
full reviews and guides if you need help deciding which deals to snag. These are the best Apple Black Friday deals you can still get today from retailers like
Amazon, Target, Best Buy and B&H Photo.
</item 43>

<item 44>
The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more (Engadget, 2023-11-27T10:00:30+00:00)
iPad Air

The base model of the latest generation of the iPad Air has an MSRP of $599 and is now down to $499, a price its hit quite a few times in 2023. This is the iPad
we recommend for most people because it’s speedy, has a great screen and works with current-generation accessories. It can act as a rather powerful
entertainment option for games and streaming, but is powerful enough to serve as a productivity device when paired up with a stylus or mouse and a keyboard. Now
that the M3 chip has been loosed upon the world, the iPad Air, which houses the M1 chip, is primed for an update, which will likely come next year (along with a
potential price increase).

iPad mini

For a more portable iPad experience, the iPad mini offers a bigger screen than your phone, but is barely as easy to carry around with you. It’s listed at $499,
but has dropped as low as $400 many times this year.
</item 44>

<item 45>
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
</item 45>

<item 46>
The best Xbox controller to buy right now (The Verge, 2023-11-21T20:04:33+00:00)
It’s an improvement on the already excellent Elite controller, with deep customization, optional rear paddle buttons, a swappable D-pad, and analog sticks that
allow you to tailor its layout to suit your play style. $140 at Walmart$140 at Microsoft

Connectivity: Xbox wireless, Bluetooth, wired / Connector type: USB-C / Mappable rear buttons: Up to four / Software customization: Yes / Power: Built-in
rechargeable

The Xbox Elite Wireless Controller Series 2 is Microsoft’s fancied-up, premium version of the stock controller. It’s also older, having been released a year
before the Xbox Series X / S consoles arrived. So while it does sport a USB-C port and wireless support, it lacks the convenient Share button found on the
latest Xbox Wireless Controller. However, it makes up for that in droves with exceptional build quality, extra buttons, user-swappable sticks and D-pad, and
lots of customization.

If you pick up an Xbox Elite controller, you instantly recognize the quality of materials put into it.
</item 46>

<item 47>
The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more (Engadget, 2023-11-27T10:00:30+00:00)
MacBook Air (13-inch, 2022, M2)

The M2-powered, 13-inch MacBook Air is on sale for $929 at B&H Photo. It’s our top recommendation for a MacBook and beat out all brands in our laptop guide. The
M2 chip is impressively fast and capable, the design is solid and the display is lovely. It’s more than capable for both work and play and should give you years
of dependable use. But, again, the M3-enabled Airs are likely on their way in 2024. So if you know you’re drawn to the newest thing, you might want to wait.

MacBook Air (13-inch, 2020, M1)

The MacBook Air with the M1 chip is a couple years old at this point, but still officially part of Apple’s laptop lineup.
</item 47>

<item 48>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
M2 MacBook Air

As for the newer, fancier 13-inch MacBook Air with M2, the base model with 256GB of storage and an eight-core GPU recently got a price drop. Along with
introducing a 15-inch M2 Air, Apple announced during this year’s WWDC that the 13-inch model now starts at $1,099 instead of $1,199. As for deals, the 13-incher
is currently selling for $899 ($200 off) at B&H Photo, Additionally, Best Buy and B&H Photo are offering an equal diacount on the upgraded configuration, which
can be had with a 512GB SSD and 10-core GPU for $1,199 ($200 off).

The M2 MacBook Air is a super slim and light laptop with a 1080p webcam that’s actually usable as well as a handy magnetic charger that frees up one of its
precious two USB-C ports.
</item 48>

<item 49>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It's only available with Tactile Quiet switches, but these are ideal if you typically work around other people and don’t want to disturb them.

This microfiber mat will keep your desk looking nice and clean. It also doubles as a giant mousepad. It only comes in black, but it's a great option if you're
looking for a desk mat that's minimalist and affordable.

Incase Hardshell Case Dots Photograph: Incase

This is currently our favorite MacBook case. We tested it with the 16-inch MacBook Pro (2021), but it’s available for a few MacBook models. It has rubberized
feet that prevent it from sliding, precise cutouts for all ports, and ventilation to keep the MacBook from overheating.

The Das Keyboard MacTigr (9/10, WIRED Recommends) is expensive, but it’s great for those in the Apple ecosystem with a dedicated Mac layout.
</item 49>

<item 50>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
Unless demand drops, Kuo said supply may remain tight until early 2024.

MacBook sales have waned this year, mainly due to "the limited M2 computing power upgrade," Kuo wrote on X. That might be a reason why Apple is willing to start
shipping M3 MacBook Pros now, even if supply is constrained. Announcing new models just a few days before reporting quarterly earnings could help give investors
confidence in the company (recent increases of Apple TV+, Apple Arcade and Apple News+ prices could play into that too).

If M3-powered models don't lead to a boost in sales, though, Kuo suggests it's more likely that a MacBook Pro with a new design will arrive in 2025.
Furthermore, Apple may try to beef up the division with a more budget-friendly MacBook that has an annual sales target of between 8 million and 10 million.

USB-C accessories

Rumors also point to Apple unveiling more USB-C accessories as it shifts away from the Lightning port.
</item 50>

<item 51>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
You can see a pattern emerging. The M3 Max outperforms the M3 on everything (no surprise there) and the M2 Ultra on several single-core tests, while often
coming within spitting distance with some of the multi-core tests.

With GeekBench 6 Metal, which tests GPU performance during complex tasks, the M3 Max scored an average of 153273. That’s a nice increase over the M2 Max’s
131883 average, while still getting trounced by the M2 Ultra’s 208584. AMD’s cards still best even the top-performing Apple chips by a good margin, with the
Radeon RX 6900XT scoring 242924.

Relative to its own family of chips, the M3 Max is doing impressive things, given that the M2 Ultra tops out at a 24-core CPU, 76-core GPU and 32-core neural
engine. The Mac Studio’s large body and airflow system are also much more equipped to dissipate heat, when compared to the MacBook Pro’s compact profile.
</item 51>

<item 52>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
This is the newer, faster version of one of our top picks in our Best Cheap Laptops guide. You're getting a newer Intel chip (13th gen i5), more RAM (16
gigabytes, a must-have with Windows 11), and a nicer IPS display all for the same price as the pick in our guide. Again, it's not a gaming rig, but it's fast
enough for everyday tasks like browsing the web and editing documents.

Photograph: Amazon

This is a good deal on an excellent controller, and the price extends to several different color options. Our favorite part about the DualSense controller is
the way the haptic feedback works. Rather than random rumbles, you'll feel directional and immersive feedback—almost like surround sound, but in a controller.

Nearly any monitor can be rotated to be a vertical monitor, and we love them for a work-from-home setup. However, the LG DualUp is made to be vertical and comes
with a wider aspect ratio (16:18) to accommodate its upright lifestyle.
</item 52>

<item 53>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
The PDP Victrix BFG Wireless Controller for PlayStation 5, PS4, and PC in its Call of Duty: Modern Warfare II-branded black-and-gold design is on sale at Amazon
for $169.99 ($30 off). I know this controller looks very Extra (and it is), but it’s a very versatile DualSense alternative if you like FPS and fighting games.
It’s got multi-stage hair triggers, and its sticks and face buttons can be swapped out for an Xbox-style layout or as a six-button fightpad.

Verge Deals on X (formerly known as Twitter) / Join more than 51,000 followers and keep up with the best daily tech deals with @vergedeals Follow us!

Wireless gaming headsets, Laptops, and PC accessories

Asus ROG Flow Z13 $ 893 $ 1300 31 % off $ 893 $ 893 $ 1300 31 % off The ROG Flow Z13 is a gaming tablet with a detachable keyboard, kickstand, and plenty of
chops for QHD gaming.
</item 53>

<item 54>
The best Xbox controller to buy right now (The Verge, 2023-11-21T20:04:33+00:00)
The new Fusion Pro 3 is significantly lighter than its predecessor, plus it has a more compact case and is $10 cheaper — offering quite the value for $79.99
(especially if it starts going on sale). The older Fusion Pro 2 is still fine, especially if you prefer detachable rear paddles to built-in buttons, but at this
point, it’s only worth it for that preference or if it’s heavily discounted.

PowerA Fusion Pro 3 $ 79 The Fusion Pro 3 is the latest version of PowerA’s affordable alternative to premium controllers like the Xbox Elite Series 2. It comes
with swappable sticks, friction rings, a compact zip-up case, and features four programmable buttons built into its rear. $79 at Amazon$80 at PowerA

The best premium Xbox controller

Xbox Elite Wireless Controller Series 2 $ 140 $ 180 22 % off $ 140 $ 140 $ 180 22 % off The Xbox Elite Series 2 is easily one of our favorite controllers at The
Verge.
</item 54>

<item 55>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
They're all included in our guide to the Best Laptops.

Apple's brand-new M3 chips are available in three different MacBook tiers—the M3, M3 Pro, and M3 Max (9/10, WIRED Recommends). They're all included in our guide
to the Best Laptops. The M3 Pro falls in the middle as far as processing power and speed goes, but it's still blazing fast. We recommend that you opt for the
Pro if you’re a serious productivity user or content creator. This deal matches the best price we've seen so far.

MacBook Air Photograph: Apple

The MacBook Air (7/10, WIRED Recommends) packs a modern design complete with a MagSafe connector so the laptop won't fly off if you accidentally trip on the
cable. Thanks to the M2 chip, it also delivers plenty of power for everyday tasks, solid battery life, and a 1080p webcam.
</item 55>

<item 56>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The Max comes in both 14-core CPU/30-core GPU and 16-core CPU/40-core GPU flavors. Again, the one I am typing this on falls into the latter camp. That also
means 48GB of memory and 1TB of storage by default. It’s a $4,000 computer, as configured. If you want to bump the memory and storage up to 128GB and 8TB,
respectively, it comes to $7,200. That’s a pretty hefty increase from the base level 14- and 16-inch models, which run $1,600 and $2,500 to start.

Apple lowered the barrier of entry for the lowest end model, but — as ever — upgrades start to add up quickly. We’re talking about a real investment here for
you — or better yet, your company. It’s certainly a system that’s likely to last you a while.
</item 56>

<item 57>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Apple MacBook Pro M3

Apple released its its new M3-powered MacBook Pros just before Halloween, but the new notebooks are already on sale at Amazon, B&H and Best Buy. The entry-level
14-inch variant with the base M3 chip, 8GB of memory and 512GB of storage is $1,449, or $150 off Apple's list price. We note in our review that the M3 chip may
quickly begin to push the limits of what 8GB of RAM can handle, though. If you can afford to step up to a model with more memory, note that a version with
Apple's M3 Pro chip and 18GB of RAM is also $200 off at $1,799. An M3 Pro version of the larger 16-inch Pro, meanwhile, is $200 off at $2,299.

Hulu Black Friday deal

Hulu’s Black Friday deal is a great one for anyone who has not subscribed to the ad-supported tier yet.
</item 57>

<item 58>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
That said, if you plan on going with a higher-spec configuration than that, you’ll likely want to spring for the M3 Pro model instead (see below).

The 14- and 16-inch MacBook Pros for late 2023 are another round of spec-bump models, much like their previous 2022 incarnations. Now starting with the M3 Pro
chip (or the speedier M3 Max) and 18GB of base RAM instead of 16GB, the new models remain targeted at creatives doing content work like video editing, photo
processing, and other graphical work. Like their predecessors and the M1 Pro generation before that, they offer MagSafe charging, three USB-C / Thunderbolt 4
ports, HDMI-out, and a full-size SD card slot, with prices starting at $1,999 for the 14-inch and $2,499 for the 16-inch.
</item 58>

<item 59>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
There were still a few during the event, but four-year-old non-proprietary chips are no longer the main point of comparison. Enough time and upgrades have
elapsed since the M1 arrived. In fact, this Friday marks three years since it was announced. Apple again,

Rendering speeds are now up to 2.5x faster than on the M1 family of chips. The CPU performance cores and efficiency cores are 30 percent and 50 percent faster
than those in M1, respectively, and the Neural Engine is 60 percent faster than the Neural Engine in the M1 family of chips.

It’s worth pointing out here that Apple’s chip numbering system isn’t as straight forward as it seems at first blush. For instance, while the M3 represents an
advance over the M2, you can still expect to see, say, the M2 Ultra to outperform the M3. Pros, Maxes and Ultras are progressively larger versions of their
baseline chip, containing more CPU, GPU and neural engine cores.
</item 59>

<item 60>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
Lenovo also offers a price guarantee refund for laptops and personal desktop computers bought from Lenovo.com within 30 days of purchase. To receive a refund,
you must contact Customer Care and show the lower price of the identical product offered on Lenovo.com. This policy doesn’t apply to doorbuster or clearance
products, however.

Lowe’s will match the price of identical, in-stock products when you present an ad, product listing, or photo from a qualifying online or local retailer,
including Amazon, Best Buy, Target, and Walmart. The product must be purchased directly from the retailer and not a third-party seller, however, and it doesn’t
apply to special sales like Black Friday and other various holiday promotions. Lowe’s also doesn’t match prices from one Lowe’s store to another. To request a
price match, head to the store, call 1-877-465-6937, or chat online.

If Microsoft lowers the price of the Xbox Series X, Microsoft could refund you the difference.
</item 60>

<item 61>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
However, you can also get a tricked-out M1 Max (10-core CPU / 32-core GPU) model with 64GB of RAM and 2TB of storage for $2,499 ($1,800 off) at B&H Photo. I
know that’s not exactly affordable, but that laptop once sold for $4,299 and still performs well in 2023.

The best Mac Mini deals

The latest Mac Mini comes in a base configuration with Apple’s M2 processor or in a more powerful configuration with the M2 Pro. It’s an excellent upgrade to
one of the most affordable yet performant desktop computers you can get (as long as gaming isn’t your priority). The M2 Mac Mini starts with 8GB of RAM and a
256GB SSD for $599, while the M2 Pro model features a superior processor plus gigabit ethernet, 16GB of RAM, and 512GB of storage for $1,299.

The latter model also features an expanded port selection, from two USB-C ports to four.
</item 61>

<item 62>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
If Microsoft lowers the price of the Xbox Series X, Microsoft could refund you the difference. Photo by Vjeran Pavic / The Verge

Now through January 31st, Microsoft will refund you the difference if it drops the price of a physical product purchased online from the Microsoft Store. The
same policy also applies if you find the product on sale at a lower price at a qualifying retailer, such as Amazon or Best Buy.

You can contact Microsoft’s Sales and Support team at 1-877-696-7786 to get a refund. If you purchase an item from a Microsoft Experience Center, you’ll have to
go back to that location with your receipt.

Newegg offers price matching on select items; these products are identified by the label “Price Match Guarantee” that appears in the price box on the right side
of the product page. This applies whether you find that lower price on Newegg or at what Newegg judges to be a major retailer.
</item 62>

<item 63>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
This thing is going to get you through your next flight, no problem.

However, if you’re flying a lot and want a MacBook travel companion, I still recommend the 15-inch Air. It continues to be the best MacBook for the vast
majority of users. The 14-inch Pro is worth looking at gaming is important — though I’m still a long way off from recommending any Macs are pure gaming
machines. The 16-inch is the pick for those who really need to up their professional game, but still need some portability. For pure desktop power, the Mac
Studio and Mac Pro with the M2 Ultra is still king — until the M3 Ultra rolls out.

To state the obvious, the 16-inch MacBook Pro with M3 Max is the most powerful Apple laptop, ever. You’ll still get more performance out of the M2 Ultra in the
Mac Studio, but for most intents and purposes, this is the desktop of laptops. It’s big, heavy, expensive and powerful.
</item 63>

<item 64>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
The front-facing camera is now on the landscape edge, which is much more convenient for video calls, though the tablet (for some reason) still only works with
the first generation Apple Pencil.

Apple M2 Mac Mini

Apple's M2-powered Mac Mini desktops are also discounted for Black Friday: The 256GB model has dropped to $500 at Best Buy, while the 512GB model is on sale for
$690 at Amazon with a clippable coupon. Though we expect an upgrade to the new M3 chip at some point over the next year, the current model is still a great
option if you want a speedy desktop that runs macOS. It earned a score of 86 in our review, and it has a headphone jack, a Gigabit Ethernet port, two USB-A
connectors, two Thunderbolt 4 USB-Cs and an HDMI 2.0 port.
</item 64>

<item 65>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
Writer Jaina Grey has had one of these for years and despite frequently losing (and finding) it, it’s an everyday carry item for her and really helps turn your
iPad into a portable workstation.

Jump to a Topic: iPads, MacBooks, Apple Watches, Beats and AirPods, Chargers, Cases and Accessories.

MacBook and Accessory Deals

Read our Best MacBook guide for more.

MacBook Pro Photograph: Apple

It's been less than a month since Apple released this MacBook Pro and it's already on sale. This M3 version replaces the 13-inch MacBook Pro, which Apple no
longer sells through its stores. It packs the same mini LED screen, 120-Hz refresh rate, brighter display, and 1080p webcam as the high-end versions. You'll
also get an HDMI, two USB-C ports with Thunderbolt 4, an SD card slot, a high-impedance headphone jack, and a MagSafe charging port.
</item 65>

<item 66>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
Mac Mini M2

The base model of Apple diminutive desktop computer, the Mac mini, is currently on sale for $100 off for Black Friday at Amazon, B&H Photo, Best Buy. It earned
an 86 in our review because it packs a lot of processing power, using the same M2 chip as the two current-model MacBook Airs — but it’s much cheaper than either
laptop. If you need a little more storage, you can get it with 512GB of SSD for $699 at B&H Photo.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.
</item 66>

<item 67>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The best MacBook Pro deals

The MacBook Pro line is once again in a transitional phase. Apple announced Macs using a new M3 processor at its “Scary Fast” event the night before Halloween,
including a new 14-inch MacBook Pro that replaces the 13-inch M2 model. That means Touch Bar Macs are finally dead, but some of those outgoing M2 models are
still available for now. You’ll find any deals on them further down, but first, let’s focus on the new M3 generation.

M3 and M3 Pro MacBook Pro

The new entry point into the MacBook Pro world is a MagSafe-equipped MacBook Pro that uses the existing 14-inch design and slightly pares it down. The 14-inch
M3 MacBook Pro has the same 3024 x 1964 resolution display and 120Hz refresh rate as its pricier siblings, and in addition to a MagSafe charging port, it has an
SD card slot and HDMI port.
</item 67>

<item 68>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
Security researchers found that the feature, introduced in iOS 14, was not working as intended.

$400 million security acquisition: At a time when the tech industry in Israel is disrupted due to political events, Palo Alto Networks is acquiring Israeli tech
company Dig Security for an estimated $400 million. This acquisition will bolster Palo Alto’s cloud security-focused Prisma business.

I guess we gotta pay up: Paying off hackers in the event of a ransomware attack is pretty common practice, but it comes with a ton of pitfalls: You are trusting
anonymous criminals to do what they promised, and you may potentially be violating U.S. sanctions laws along the way.

What’s brewing on the silicon farm?

Apple’s “Scary Fast” October event saw the announcement of their new M3 chip lineup that includes M3, M3 Pro, and M3 Max. These chips, which were the highlight
of the event, will be integrated into the new 24-inch iMac and the new MacBook Pros.
</item 68>

<item 69>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
This thing is a beast — in pretty much every sense. It’s a big, heavy, powerful system, with a bright display, massive battery life and impressive power under
the hood. After opening the box, I immediately ported all the contents over from the 15-inch Air I’ve been using daily for several months now, but still, the
16-inch MacBook Pro feels like a different species.

Each has their place. I’m hopping on another cross-country flight tomorrow, and I’m planning to leave the 16-inch Pro at home. This is due partially to an
ongoing back injury, but frankly, for as good as the Pro’s battery life is (very good, for the record), the Air is a much better travel companion. It’s thin,
it’s light, it has more than enough processing power for 95% of the things 95% of people need, and it won’t wrestle you for the arm rest.
</item 69>

<item 70>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
I will say, however, that I have so far only triggered the fan during benchmarking. The Cinebench test was run with the system in high-power mode, a setting
that prioritizes performance over battery life. Low-power mode, naturally, does the inverse, while automatic (the default setting) does its best to split the
difference.

And Finally

Honestly, though, the battery might be the most exceptional bit here. The first thing I did upon receiving the system (after setup and charging) was a video
rundown — far and away the most lengthy part of testing. I looped a single movie (Bill & Ted Face the Music, for the record) in Apple TV, with brightness and
volume at their default settings. I kicked it off at 4:45PM on Tuesday, and it ran until 7:30 the following evening. That’s a runtime of 26 hours, 45 minutes.
This thing is going to get you through your next flight, no problem.
</item 70>

<item 71>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
He found the iMac to be lacking and not necessarily worth the upgrade from the 2021 model, excepting the M3 chip, which brings “impressive” performance gains
over the already-powerful M1. As for the M3 Max MacBook Pro, Brian reports that, at $2,500 (plus some pricey add-ons), it successfully splits the difference
between the Mac Studio and MacBook Air.

Mozilla bets on a decentralized future: Sarah spoke with Mozilla senior director of content Carolyn O’Hara, who outlined Mozilla’s strategy where it concerns
the “fediverse” — a collection of decentralized social networking applications, like Mastodon, that communicate with one another over the ActivityPub protocol.
The idea, O’Hara said, is to rethink social networking from the ground up.
</item 71>

<item 72>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
They carry a list price of $169 but often sell for closer to $150, so this is $10 off the usual. These non-Pro AirPods came out in 2021 and got a review score
of 88 from us. They pair quickly with iPhones and are a good complement to Siri’s helpfulness. They don’t have ANC however — you’ll need to go for the AirPods
Pro for that feature — but they do have Adaptive EQ which monitors and adapts sound in real time. The fit is comfortable and they’re water-resistant so they can
handle workouts or the weather.

MacBook Black Friday Day deals

MacBook Pro (14-inch, 2023, M3)

The base model, 14-inch MacBook Pro with the new M3 chip just came out in October but is already$150 off for Black Friday. After testing the new chip, we think
the base model with 8GB of RAM may quickly get bogged down.
</item 72>

<item 73>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
Apple fans love a new color. In the case of the MacBook line, it could be attributable to the fact that there’s little variation in the laptops’ aluminum
design. The last few generations have traded almost exclusively in space gray and silver (dark and light gray, essentially). The new Airs opened this up a bit,
however, with the addition of starlight (a subtle light-gold) and midnight (a much darker gray). The space black is a nice addition. It gives off a nice glow
when the light hits it. The biggest thing here, however, is the fingerprint factor. The Airs I’ve tested were silver and starlight, so I can’t really speak to
this directly, but the midnight Air is a notorious fingerprint magnet. There are pages and pages of conversations about it on Reddit, including “tips” for a
fingerprint-free midnight Mac. Have you tried typing in gloves? Not easy.
</item 73>

<item 74>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The triangle computer shader makes processing complex geometrical images more efficient. That, in turn, allows for higher-quality graphics and more scene
detail.

The results are impressive. I’ve been spending a bit of time with the Soulslike title, Lies of P. It’s basically Pinocchio with more swords. It’s of a small –
but increasing – number of day-and-date releases for macOS. A couple of years ago, it would have been inconceivable for the operating system to get the game the
same day as PS5, Xbox Series X/S and Windows. That said, the macOS gaming library still has a long way to go.

Indeed, the gameplay was smooth and detailed – something that also would have seemed impossible not all that long ago. Sony’s PS5 DualShock controller connects
seamlessly and makes for a fun portable gaming experience.

One thing I noticed during last week’s presentation is that Apple is leaning less heavily on Intel comparisons.
</item 74>

<item 75>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The latter model also features an expanded port selection, from two USB-C ports to four. It’s almost like getting an M2 Pro-powered MacBook Pro 14 but in
desktop form. However, keep in mind that buying any Mac Mini means you have to provide your own mouse, keyboard, and monitor.

Right now, the base M2 Mac Mini is selling for a new all-time low of $479 ($120 off) at Amazon and B&H Photo, while the M2 Pro version is selling for a
respectable $1,149 ($150 off) at B&H Photo. The M2 Mac Mini remains a great value for a desktop computer, with the M2 Pro version functioning like a baby Mac
Studio for much less.

Related The best thing about my Mac Studio is its mustache
</item 75>

<item 76>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
It clears hard and carpeted floors well, and it automatically returns to its base once a cleaning job is finished so it can recharge.

Razer Basilisk V3

The Razer Basilisk V3 is down to $44, which is a few bucks more than its all-time low but still $6 less than its usual street price. This is the “best for most”
pick in our guide to the best gaming mouse, as its sturdy shape should be comfortable for each grip type and all but the smallest hands. (Provided you’re a
righty.) It has 11 customizable buttons, including a “sensitivity clutch” for more precise aiming, and its thumb wheel supports both left-right tilt and a
faster free-spin mode. That wheel is somewhat noisy, and the whole thing is too heavy for competitive-minded players, but it still tracks accurately and glides
smoothly.

Samsung Pro Plus microSD

The Samsung Pro Plus microSD card is on sale for $20 right now.
</item 76>

<item 77>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It also has cherry MX Red switches (that don’t get too loud), a two-port USB-C hub, and a high-quality all-metal build.

Photograph: Keychron

This is WIRED reviews editor Julian Chokkattu's current daily keyboard. You can use it wired or wirelessly via Bluetooth and it comes in several colors and
switches to pick from (it’s also RGB-backlit). The red switches aren’t very loud but are still clicky, and there’s a great knob for volume control. It’s
hot-swappable too, so you can change the switches whenever you’d like.

If you prefer low-profile keys, the K3 Pro is a solid choice. The keys have a tactile click, RGB lighting, and you’ll get around four to five days of battery
life (with the ability to charge it via the USB-C charging port). You’ll also get support for Via too, if you want additional software customization.
</item 77>

<item 78>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The redesigned M2 version of the MacBook Air has been with us for a bit — there’s now even a 15-inch M2 Air — but the 2020 version with an M1 processor and
fanless design remains in the lineup as the budget option. For many people, the M1 Air still ticks the right boxes when it comes to performance and price, even
if it’s long enough in the tooth to have been fully dethroned in our guide to the best laptops.

The base MacBook Air with the M1 chip comes with 8GB of RAM and 256GB of storage. It typically sells for $999, but Amazon and Best Buy are currently discounting
it to $749.99 ($250 off), matching its lowest price. The M1 Air may be a few years old now but it’s still hard to beat in terms of value — especially when it
comes to everyday performance and battery life.
</item 78>

<item 79>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
Around that, you’ll see a series of small screws that signal the company’s newfound push into repairability spurred on by the broader call for right to repair
legislation. There is also a heat vent on either side.

The Pros are a good bit thicker than the Air. The 14- and 16-inch are 0.61 and 0.66 inches tall, respectively, versus the 15-inch Air’s 0.45 inches. This is
due, in no small part to the Pros’ inclusion of a fan and airflow system. You’re unlikely to trigger the fan during most everyday operations, but more complex
tasks like 3D rendering require a lot of compute power, so it’s there when you need it. Meanwhile, the 14-inch Pro (3.4 pounds) weighs just a hair more than the
15-inch Air (3.3 pounds). At 4.7/4.8 pounds (for the M3 Pro and Max), there’s a pronounced increase with the 16-inch Pro.
</item 79>

<item 80>
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
$245 at Amazon$250 at Best Buy

Seagate Storage Expansion Card (1TB) $ 130 $ 220 41 % off $ 130 $ 130 $ 220 41 % off Proprietary SSD expansion for the Xbox Series X / S consoles. The
plug-and-play drives are designed to be as fast as the Xbox internal SSD and are sold in 512GB, 1TB, and 2TB configurations. $130 at Best Buy

The best TV and streaming device deals

2022 Samsung The Frame TV (55-inch) $ 1000 $ 1500 33 % off $ 1000 $ 1000 $ 1500 33 % off Samsung’s art-inspired 4K TV features a QLED display and an anti-glare
matte display, which provides artwork and photos with qualities that resemble an actual canvas.
</item 80>

<item 81>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
The best (and only) deal available right now on the creator-friendly laptop is at Amazon, where you can grab the base M2 Pro model for $2,299.99 ($200 off) with
512GB storage. That’s not great, but hopefully we see better deals before the laptop is no longer available new.

M1 Pro and M1 Max MacBook Pro

For now, the older 2021 MacBook Pros can occasionally be had for some steep discounts that actually make a two-generation-old laptop still worth considering.
You’re unlikely to find any base models still kicking around, but if you want a souped-up configuration on the cheap, you may be in luck.

The 16-inch MacBook Pro from 2021 with 16GB of RAM and an expanded 1TB of storage is available at B&H Photo for $1,799 ($900 off).
</item 81>

<item 82>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
It's $200 off at B&H Photo. We gave it a 96 in our review and liked how the bigger screen let you get the most out of the M2 processor.

Now that the M3 MacBook Pros have arrived, we’re guessing Apple will come out with M3-enabled Air models in 2024. Still the M2 has plenty of processing power
and will serve you well as both a casual machine for playing games and watching shows, but can also hold its own with productivity tasks.

MacBook Air (13-inch, 2022, M2)

The M2-powered, 13-inch MacBook Air is on sale for $929 at B&H Photo. It’s our top recommendation for a MacBook and beat out all brands in our laptop guide. The
M2 chip is impressively fast and capable, the design is solid and the display is lovely. It’s more than capable for both work and play and should give you years
of dependable use.
</item 82>

<item 83>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
It’s a lot to get through, as always — so we won’t delay. But first, a reminder to sign up here to receive WiR in your inbox every Saturday if you haven’t
already done so.

Most read

OpenAI throws a dev day: OpenAI hosted its first-ever developer conference on Monday, and the company had a lot to talk about. Some of the more notable items
announced were tools to create custom “GPTs” (i.e., domain-specific chatbots), new text-to-speech models, an API for the text-to-image model DALL-E 3, and an
improved version of OpenAI’s flagship model, GPT-4, called GPT-4 Turbo.

Mac attack: Brian reviewed Apple’s new 16-inch M3 Max MacBook Pro and the M3 iMac 24-inch.
</item 83>

<item 84>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
MacBook and Accessory Deals

Read our Best MacBooks guide for more info and other options.

Apple MacBook Air Photograph: Apple

The M1-powered MacBook Air (9/10, WIRED Recommends) is our favorite budget MacBook. It's a few years old by now, but it's still one of the most powerful laptops
you can buy for this price. It's fast and light and can make quick work of simple office tasks. It also has an improved keyboard (the one that doesn't break).
The Air can handle some light video editing if the need arises, but if you need more power, spring for one of the newer, beefier options.

MacBook Pro Photograph: Apple

We really only recommend this MacBook Pro (7/10, WIRED Review) if you want the touch bar.
</item 84>

<item 85>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
But if you want to buy new and you’re looking to save whatever you can, here are the best MacBook Air, MacBook Pro, and Mac Mini deals available right now.

A note about discounted Intel-based Macs At this point, we don’t recommend buying any older Intel-based Macs. Retailers may try to tempt you with very low
prices on leftover new or refurbished stock, but there’s a reason nobody wants them anymore. They’re completely outclassed by all the newer models, unless you
have some super-niche Windows Boot Camp needs — and even then, consider what Windows apps you really need and if they work with Parallels.

The best MacBook Air deals

M1 MacBook Air

The MacBook Air is Apple’s entry-level laptop. It’s best suited for typical productivity work, with a comfortable keyboard, an excellent trackpad, and all-day
battery life.
</item 85>

<item 86>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
That's a $30 discount. Both the USB-C and Lightning models are on sale; the latter is compatible with Apple's iPhone 15 series as well as numerous Android
phones. We gave the Backbone a positive review last year, and we recommend it in our guide to the best gaming handhelds: It's more cramped than a traditional
gamepad, naturally, but it has all the requisite buttons to play modern games, it's extremely easy to set up and it works well for remote streaming from your
console or PC. If nothing else, it's a much better value than the PlayStation Portal.

PlayStation, Xbox and Switch video game sales

If you're all set on the hardware front, a truckload of PlayStation 5, Xbox Series X/S and Nintendo Switch games have dropped in price for Black Friday too.
Both Sony and Microsoft have rolled out holiday sales on their respective digital storefronts, while a number of physical games are down to new lows at Amazon,
Best Buy and others.
</item 86>

<item 87>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Valve will be selling three models:

$649 - 1 TB OLED (up from 512 GB)

$529 - 512 GB OLED (up from 256 GB)

$399 - 256 GB non-OLED Steam Deck (up from 64 GB)

Plus, for a limited time, the company will offer a special edition 1 TB Steam Deck OLED made with a translucent shell for $679 — only in the U.S. and Canada.
Though let’s be real: This thing will sell like hotcakes and Valve will reconsider the definition of “limited supply.” Have I considered ordering one of these
limited editions? Yes. But I am also practicing restraint. Like, how many Steam Decks should one human have?

I suppose that’s the question, really. If you already have a Steam Deck, is the OLED model technically an incremental improvement? Yes. But you might find that
the screen and the battery justify the double dip.
</item 87>

<item 88>
What we expect at Apple's Scary Fast event: M3-powered MacBook Pros and new iMacs (Engadget, 2023-10-26T19:15:49+00:00)
That could all start with upgraded chipsets. It's been nearly 17 months since Apple's M2 system on a chip (SoC) debuted. With many manufacturers having an
annual cadence for their chipsets, it may be time for Apple to move onto the next generation. (No, shoving two M2 Max chips together isn't quite the same
thing.) For what it's worth, the M2 series debuted 19 months after the M1. So, if Apple announces the M3 at the Scary Fast event, the latest SoC would be
arriving after a similar period of time.

Apple analyst Ming-Chi Kuo has suggested that "M3 series MacBook Pro" will be the focus of the event, indicating that the company will have more than one new
SoC to reveal. As such, there's a chance that we could see an M3 Pro and maybe even an M3 Max during the Scary Fast showcase.
</item 88>

<item 89>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
In fact, it shouldn’t go unremarked upon that the arrival of the new 14-inch signaled the merciful end to that particular technology, as the last Touch Bar Mac
— the 13-inch Pro — was replaced by the new 14-inch model.

Design

I would say, more than anything, the thing I miss moving from the MacBook Pro to the Air is the ports. True story: At last year’s CES, I had to locate an Apple
Store in the Venetian because my external SD reader failed (I won’t say which TechCrunch staff member laughed at me in that predicament, but they know who they
are). I understand that an SD reader is kind of a niche need in these days of everyone shooting everything on their, so I can at least understand Apple’s
justification for keeping it to the Pro system (the new iMac doesn’t have one, either).

Ports are a different story. More ports is better. It’s simple math. People can’t get enough of the things.
</item 89>

<item 90>
The best Xbox controller to buy right now (The Verge, 2023-11-21T20:04:33+00:00)
Where the standard controller uses an unassuming plastic build, the Elite mixes higher-quality plastic with rubberized grips and bits of metal. Combine this
with its built-in rechargeable battery, and the Elite is noticeably heavier than its cheaper brethren. It also comes with a charging dock and zip-up case with
passthrough charging.

The Xbox Elite Series 2 is made from premium materials but has a nice, subdued aesthetic. The metal, dish-like D-pad is a small work of art. Sadly, however,
it’s a far cry from the best D-pad around. The four removable paddles on the rear sit beneath the trigger locks and charging dock pins.

As excellent as the fit and finish are, the key components that may give you a competitive edge are the hair-trigger locks and rear paddle buttons. In theory,
the hair trigger gets shots off faster when playing shooters, and the rear paddles allow you to jump, crouch, reload, etc., without taking your thumb off the
right stick for aiming.
</item 90>

<item 91>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
But it differs by having a larger, 15.3-inch notched screen with 2880 x 1864 resolution, six speakers instead of four, a 10-core GPU that costs extra on the
13-inch, and a higher starting price of $1,299. Bumping up to a larger, faster 512GB of storage will run you $1,499.

Surprisingly, the 15-inch M2 Air started receiving some discounts even before it shipped, which let early adopters save between $50 and $100. As of right now,
you can get it $300 off by ordering it at Amazon or Best Buy, where it’s currently selling for $999. The expanded 512GB model is also $300 off at Best Buy,
selling for $1,199.

The best MacBook Pro deals

The MacBook Pro line is once again in a transitional phase.
</item 91>

<item 92>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
It rarely goes on sale.

This is our favorite external keyboard in our guide to Best MacBook Accessories. It's a great option if you're a fan of the Magic Keyboard and are looking to
fully replicate your MacBook setup. However, this version is fairly bare-bones—it doesn't come with a Touch ID button or the same function row keys as the
M-series MacBooks (that model isn't on sale, sadly). But it does come with the standard keys like playback controls, a Mission Control key, as well as a number
pad. This is also the lowest price we've tracked for this keyboard, so far.

The Studio Display (9/10, WIRED Recommends) has a spacious 27-inch display that's perfect for juggling multiple apps at the same time and a 5K resolution that's
stunningly sharp (there's no HDR though, so colors look slightly more contrasty and saturated compared to the neutral tones on Apple's Pro Display XDR).
</item 92>

<item 93>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
iPad Air

The base model of the latest generation of the iPad Air has an MSRP of $599 and is now down to $499, a price its hit quite a few times in 2023. This is the iPad
we recommend for most people because it’s speedy, has a great screen and works with current-generation accessories. It can act as a rather powerful
entertainment option for games and streaming, but is powerful enough to serve as a productivity device when paired up with a stylus or mouse and a keyboard. Now
that the M3 chip has been loosed upon the world, the iPad Air, which houses the M1 chip, is primed for an update, which will likely come next year (along with a
potential price increase).

iPad mini

For a more portable iPad experience, the iPad mini offers a bigger screen than your phone, but is barely as easy to carry around with you. It’s listed at $499,
but has dropped as low as $400 many times this year.
</item 93>

<item 94>
The best controllers for PC gaming (Polygon, 2023-11-17T13:15:48+00:00)
If you don’t want to change out batteries, we recommend this $44.99 8BitDo dual controller charging dock, which includes two rechargeable battery packs, enough
for two controllers.

Microsoft’s Xbox Elite Series 2 builds on the excellent foundation of Microsoft’s Xbox controller with features that enthusiast gamers may appreciate. It
features swappable stick caps and directional pads, adjustable tension for its analog sticks, and customizable back paddles. It also ships in a nice zip-up
case. If you want more buttons and tweakable settings in a controller, this might be worth its $178.99 price. This model is compatible with Xbox and PC.

If you don’t need all of the extra accessories that come with the Elite Series 2, Microsoft makes a cheaper Elite Series 2 Core. If you decide later on that you
want the paddles, zip-up case, and swappable sticks, you can buy this third-party kit on Amazon for $22.99.
</item 94>

<item 95>
The best deals on MacBooks right now (The Verge, 2023-10-26T18:17:19+00:00)
However, it starts with just 8GB of RAM and lacks the third USB-C / Thunderbolt port found on the M3 Pro and M3 Max models (as well as Thunderbolt 4 speeds).

The M3 MacBook Pro may be a bit of an odd middle child in some ways, but it’s still a very good laptop — especially if you can get it on a deal that puts more
price distance between it and the M3 Pro version. We’ve already seen the base model with 512GB of storage on sale for $150 off during Black Friday and Cyber
Monday, and right now, it’s actually available for an even better $1,399 ($200 off) at Amazon, Best Buy, and B&H Photo. The 1TB configuration is also selling
for its all-time low of $1,599 ($200 off) at Best Buy and B&H Photo, while the configuration with base storage and 16GB of RAM is selling for $1,599 ($200 off)
at B&H Photo.
</item 95>

<item 96>
The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more (Engadget, 2023-11-27T10:00:30+00:00)
Read our review of the new MacBook Pros with the M3 chip

MacBook Air (15-inch, 2023, M2)

The 15-inch MacBook Air arrived in June and added a little more screen real estate to an already great machine. It’s down to $1,049 for Cyber Monday at Amazon
and Best Buy. That’s the same as the all-time low it’s hit a few times these past couple of months. In addition, it's $200 off at B&H Photo. We gave it a 96 in
our review and liked how the bigger screen let you get the most out of the M2 processor.

Now that the M3 MacBook Pros have arrived, we’re guessing Apple will come out with M3-enabled Air models in 2024. Still the M2 has plenty of processing power
and will serve you well as both a casual machine for playing games and watching shows, but can also hold its own with productivity tasks.
</item 96>

<item 97>
The best Apple Cyber Monday deals of 2023: Save on AirPods, iPads, Apple Watches and more (Engadget, 2023-11-27T10:00:30+00:00)
Check out our guide to iPads here.

MacBook Cyber Monday deals

MacBook Pro (14-inch, 2023, M3)

The base model, 14-inch MacBook Pro with the new M3 chip just came out in October but is $150 off for Cyber Monday. After testing the new chip, we think the
base model with 8GB of RAM may quickly get bogged down. If it's within your budget, we recommend the model with 16GB of RAM, which is $150 off at B&H photo. And
if you upgrade to a model with the M3 Pro chip for an even speedier experience — you'll get a $200 discount as it's down to $2,199 at B&H Photo. We gave the new
M3 MacBook Pros an appreciative review and named them the best option for creatives in our buyer’s guide.
</item 97>

<item 98>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
We recommend the M3 for anyone who wants some of the Pro features but doesn't want to drop almost $2,000 to get it.

This MacBook Pro (9/10, WIRED Recommends) is less than a year old, but we only recommend it if you can find it for under $1,800. It packs all the same features
as the latest version and the M2 Pro is more than capable of handling graphically demanding and intensive tasks. This is also the lowest price we've tracked for
this specific configuration.

This M3 Pro MacBook Pro is similar to the other model that's on sale. It just has a larger screen, a more powerful CPU, and slightly better graphics. Apple's
brand-new M3 chips are available in three different MacBook tiers—the M3, M3 Pro, and M3 Max (9/10, WIRED Recommends). They're all included in our guide to the
Best Laptops.
</item 98>

<item 99>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Microsoft Surface Pro 9

A configuration of Microsoft's Surface Pro 9 with an Intel Core i5-1235U processor, 16GB of RAM, a 256GB SSD and Microsoft's Surface Pro Keyboard is on sale for
$1,000 at Best Buy. That's $540 off Microsoft's list price. If you don't need the keyboard, a variant with a faster Core i7-1255U chip is down to a new low of
$1,097 at Amazon, Microsoft, B&H and other retailers. That's roughly $200 off the config's usual street price. Another Core i7 model with 512GB of storage is on
sale for $300 more.

In any event, the Surface Pro 9 tops our guide to the best 2-in-1 tablets. While its 12th-gen chip is about to be two generations old, this is still the device
to get if you want the functionality of a laptop in the slim, touch-enabled design of a tablet.
</item 99>

<item 100>
The best Xbox controller to buy right now (The Verge, 2023-11-21T20:04:33+00:00)
It makes a great PC controller, too.

But while the de facto Xbox gamepad strikes the right balance of quality, comfort, versatility, and price, there are several alternatives worth cross-shopping
if you have specific needs — ranging from high-end options like the Xbox Elite Series 2 and Scuf Instinct Pro to quality budget picks like the PowerA Enhanced
Wired Controller.

What we’re looking for Connectivity Is it wireless? Does it support Bluetooth? Or is it wired only? If so, is the cable replaceable? Quality Does it feel cheap
or nicely made? Any glitches or reliability issues while testing? Extra features Does it have handy extras like remappable buttons, or extra ones? Customization
Can you remap buttons or tune it with custom software profiles? Are there fun cosmetic flares to its design? Value Is it priced affordably, about average, or
does it come at a steep premium?
</item 100>
=== END ITEM id=103fa34e289d4555 ===

=== ITEM id=29e119b56c998c82 ===
Gold sentences and facts:
Sentence 0: Altman’s exit “is indeed shocking as he has been the face of” generative AI technology, said Gartner analyst Arun Chandrasekaran.
  Fact 0.0: Altman's exit is indeed shocking
  Fact 0.1: Altman has been the face of generative AI technology
  Fact 0.2: Arun Chandrasekaran, a Gartner analyst, said this
Sentence 1: Altman has been telling investors that he’s planning to launch a new venture, according to The Information.
  Fact 1.0: Altman has been telling investors that he is planning to launch a new venture
  Fact 1.1: The Information is the source of this report
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory is based on the board's language
  Fact 2.1: The prevailing theory is based on the way giant tech companies work
  Fact 2.2: The prevailing theory is floating around right now

Retrieved items:
<item 1>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
That's nice and all, but the best way to appreciate the fans would be to not price out the loyal supporters from season tickets next year. Instead, Inter Miami
has jacked 2024 membership rates to put the club amongst some of the most expensive tickets in not just MLS, but the world.

While there will certainly be folks who can afford those high-demand tickets, many of those individuals won't be the same fans who have supported the club from
its inception in 2020. This means Inter Miami have chosen to secure maximum profit at the expense of its fanbase, and a high turnover amongst the regular
in-person support from this season to next is likely.
</item 1>

<item 2>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
A reminder, this match officially kicks off at 7:39 p.m. ET so there will be a slight delay from the advertised 7:30 p.m. time.

Inter Miami fan news: the 5 supporters’ groups known as La Familia threatened to stage a protest this game re: ticket price hike for 2024, but team offered to
meet w/them next week, so protest off at least until next game. @MiamiHerald @HeraldSports @InterMiamiCF #InterMiamiCF — Michelle Kaufman (@kaufsports) October
7, 2023

30 mins to kickoff: Lionel Messi's availability for Inter Miami is a welcome sight for the club, as they take the field today with their backs against the wall.
Inter Miami must win tonight to ensure they remain alive in the playoff race. A draw or a loss, however, would see them vulnerable to outside results.
</item 2>

<item 3>
Inter Miami friendly matches for 2024 preseason: Lionel Messi to play Newell's Old Boys, China tour cancelled (Sporting News, 2023-11-30T21:41:00+00:00)
With the 2023 MLS season closing earlier than expected for Inter Miami, eliminated from playoff contention, Lionel Messi will have significant time off before
the new campaign begins in February 2024.

With every new season come preseason friendlies, as the players hope to ramp up their match fitness prior to the start of the coming campaign. Players need game
time to build their bodies to the point where they can play full 90-minute matches, especially when they are contesting multiple games a week.

Inter Miami's preseason will be watched with extra observation as Messi draws considerable attention wherever he goes, home or abroad. The Argentina star had
looked like being part of a tour of China, although that's no longer the case, while the prospect of facing a familiar side from home is a tantalising one.

The Sporting News has compiled all the known preseason matches either confirmed or reported prior to the start of Inter Miami's 2024 season.
</item 3>

<item 4>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Inter Miami season tickets: Prices for 2024

The pricing structure for 2024 represents an enormous increase over the 2023 costs. The cheapest season ticket in 2023 was $485, but that will jump all the way
to $884, which is an increase of 78.8%. There is a discount for fans who renew by October 7.

Meanwhile, a season ticket in the lower-level midfield club area, which includes upscale food and beverage options, cost $3,600 in 2023 but will now run $7,650
for the 2024 season, an increase of 113%, more than doubling year-over-year.

Inter Miami 2024 season ticket prices

Prices below courtesy of Inter Miami season ticketing release via the Miami Herald on September 28, 2023. Sections 2 through B are considered "premium" seating,
while C through L are considered "standard."
</item 4>

<item 5>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Code Section Price 2 Mid Loge $45,900 3 North Loge $42,840 4 South Loge $42,840 5 Midfield Club $7,650 6 East Field Seats $13,005 A Northwest Prime $5,440 B
Northwest Midfield $4,420 C West Sideline $2,720 D West Reserved $2,380 E East Midfield $2,635 F East Upper Midfield $2,125 G East Sideline $1,955 H East
Reserved $1,360 I North Goaline $1,054 J South Goaline $1,020 K South Stand $867 L North Stand Reserved $884

Inter Miami season tickets among most expensive in world for 2024

With the Inter Miami season-ticket cost increasing by a huge margin over its own pricing model from 2023 to 2024, the club now boasts some of the most expensive
seats in world football, also rivaling that of other major sports in the U.S.
</item 5>

<item 6>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
Inter Miami's home field advantage tonight will be significantly hampered.

Not sure how much can be appreciated via the photo, but La Familia — Inter Miami’s supporters’ groups — are mostly sitting here before kickoff.



No drums, chanting, or enthusiasm in what is a protest to the increased price of season tickets for 2024.#InterMiamiCF #Messi𓃵 pic.twitter.com/YaP4ivlX0P —
Franco Panizo (@FrancoPanizo) October 7, 2023

15 mins to kickoff: Interestingly, while tonight's match should kick off without any negative attention, it seems the club just managed to get off the ground
without a hitch. According to Michelle Kaufman of the Miami Herald, the supporter groups had planned a protest regarding 2024 season ticket prices, but the club
offered to meet with them this coming week so they called it off until that can take place.

A reminder, this match officially kicks off at 7:39 p.m.
</item 6>

<item 7>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Meanwhile, another nearby rival, Atlanta United, will feature its cheapest season ticket for $605 in 2024, or just $525 for founding members.

Inter Miami season-ticket glitch causes scare

Already, price increases have Inter Miami season ticket holders on edge, but a computer glitch saw supporters fed an even bigger scare.

Mike Ryan, a Miami local who regularly appears on the radio program The Dan LeBatard Show, posted on social media his shock when he received a 2024 renewal for
$30,600 for his two seats, which in 2023 cost just $6,802. Ryan then posted a follow-up a few hours later that stated he was contacted by Inter Miami
representatives who indicated that pricing was incorrect thanks to a computer glitch that sent supporters quotes for additional seats they did not actually
have.
</item 7>

<item 8>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
Lionel Messi's arrival at Inter Miami has brought with it a craze for fans to get a glimpse of one of the sport's most talented and legendary players of all
time.

Along with that skyrocketing attention comes a leap in prices, and sadly the club seems to be passing that cost on to the supporters.

In 2024, Inter Miami will sport some of the world's most expensive tickets, rivaling the prices of Champions League giants such as Barcelona, Arsenal, and other
massive clubs around Europe, while dwarfing other MLS ticket packages.

While Fox Sports reported the club has an extensive waiting list for 2024 season tickets, the massive increase in price could push out many of the team's most
dedicated supporters who have been with the team since its inception in 2020, causing massive turnover in the fanbase.

A reminder that for most clubs, season-ticket packages include only home league matches, and while members usually get first access to purchase tickets for
knockout competitions, those all come at an additional cost.
</item 8>

<item 9>
Inter Miami season tickets: Lionel Messi demand causes 2024 prices soar to most expensive in world (Sporting News, 2023-10-04T22:12:56+00:00)
The most expensive premium packages at Inter Miami in 2024, which will run supporters over $45,000, dwarf any of the most luxurious tickets in the Premier
League. The most expensive season ticket in the EPL can be found at Fulham, where the Craven Cottage luxury booths are valued at $3,620. Barcelona's most
expensive season ticket is $1,826 for the full year.

The National Football League (NFL), the most popular sport in the United States by a massive margin, boasted an average ticket price of $112 in 2022, while the
average NBA ticket price was $94 that year.

These prices also dwarf many other clubs in Major League Soccer. Orlando City SC, one of Inter Miami's chief in-state rivals, have their cheapest season ticket
available for just $324 in 2024, over half the cost of any Inter Miami season ticket, while their most expensive premium ticket costs $2,898.
</item 9>

<item 10>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
MORE: Inter Miami season tickets for 2024 season see unprecedented jump in price

(Also, you'll notice a photo of Lionel Messi's jersey hanging in his locker in that tweet below, a pretty good indication that he'll return tonight)

Scene is set for Fan Appreciation Night 🏟️#MIAvCIN | 7:30 PM ET pic.twitter.com/9I07uq9hGW — Inter Miami CF (@InterMiamiCF) October 7, 2023

2 hours to kickoff: A report from well-connected Argentine journalist Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for
this match in some capacity. Edul reported that "the idea is for Leo Messi to get minutes when Inter Miami plays FC Cincinnati".

That makes it sound like Messi will only be available off the bench, but it's better than nothing to have a game-changer like him available at all.
</item 10>
=== END ITEM id=29e119b56c998c82 ===

=== ITEM id=b1fcfd084ec0e7ff ===
Gold sentences and facts:
Sentence 0: These delays may lead to bets being refunded depending on the circumstances and the specific rules of your sportsbook.
  Fact 0.0: Delays may lead to bets being refunded.
  Fact 0.1: Whether bets are refunded depends on the circumstances and the specific rules of the sportsbook.
Sentence 1: Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like an injury or roster move.
  Fact 1.0: Every sportsbook adjusts when a betting line moves or odds change for a legitimate reason.
  Fact 1.1: Examples of legitimate reasons include an injury or a roster move.
Sentence 2: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 2.0: Sportsbooks are also gathering this information.
  Fact 2.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.

Retrieved items:

=== END ITEM id=b1fcfd084ec0e7ff ===

=== ITEM id=13bed2ef19630bc1 ===
Gold sentences and facts:
Sentence 0: Another user said that no one can carry out such a major attack against Israel without intelligence support.
  Fact 0.0: Another user said that no one can carry out such a major attack against Israel without intelligence support.
Sentence 1: Israel stopped allowing deliveries of food, fuel and other supplies to Gaza’s 2.3 million residents on Oct. 10, 2023, and is reportedly preparing for a ground invasion.
  Fact 1.0: Israel stopped allowing deliveries of food, fuel and other supplies to Gaza's 2.3 million residents on Oct. 10, 2023.
  Fact 1.1: Israel is reportedly preparing for a ground invasion.

Retrieved items:

=== END ITEM id=13bed2ef19630bc1 ===

=== ITEM id=7a648a62ad3e8581 ===
Gold sentences and facts:
Sentence 0: He is helping to usher in a youth movement up front for Argentina, with Julian Alvarez and Alejandro Garnacho considered the future of the forward line.
  Fact 0.0: He is helping to usher in a youth movement up front for Argentina.
  Fact 0.1: Julian Alvarez is considered the future of the forward line.
  Fact 0.2: Alejandro Garnacho is considered the future of the forward line.
Sentence 1: 2 hours to kickoff: A report from well-connected Argentine journalist Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for this match in some capacity.
  Fact 1.0: The kickoff is 2 hours away.
  Fact 1.1: Argentine journalist Gaston Edul is well-connected.
  Fact 1.2: A report from Gaston Edul last night indicated that Inter Miami are likely to have Lionel Messi back for this match in some capacity.

Retrieved items:
<item 1>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Timeline of Sam Altman’s firing from OpenAI

November 29

Microsoft gains a board observer

Microsoft will gain representation on the new initial board of directors in the form of a non-voting observer, OpenAI announced. It wasn’t immediately clear who
this observer might be — only that they won’t have an official vote in board business.

November 21

Sam Altman, OpenAI reach agreement on return as CEO and ‘initial’ new board

In a sudden late announcement, OpenAI revealed that it and Altman “have reached an agreement in principle” for him to return as the company’s CEO. In addition
to Altman’s return, its new “initial” board will include former Salesforce chief executive Bret Taylor, former US Secretary of the Treasury Larry Summers and
Quora founder Adam D’Angelo.

We have reached an agreement in principle for Sam Altman to return to OpenAI as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam
D'Angelo.
</item 1>

<item 2>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
The Information, speaking to three sources formerly with the company, reports that they no longer expect the sale — led by Thrive Capital — to happen, or, if it
does, to come with a lesser valuation because of the recent turn of events.

Altman planning new venture

Altman has been telling investors that he’s planning to launch a new venture, according to The Information. Brockman is expected to join the effort — whatever
form it takes. (Possibly an AI chip startup.)

i love you all. today was a weird experience in many ways. but one unexpected one is that it has been sorta like reading your own eulogy while you’re still
alive. the outpouring of love is awesome. one takeaway: go tell your friends how great you think they are.
</item 2>

<item 3>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
Later Monday, that number climbed to over 650.

Altman and Brockman considering return

As reported by The Verge, Altman’s move to Microsoft isn’t a done deal — and both Altman and Brockman are still open to returning to OpenAI. That is, if the
remaining board members who initially fired him step aside.

OpenAI board considers merger

OpenAI’s board of directors approached Dario Amodei, the co-founder and CEO of rival large-language model developer Anthropic, about a potential merger of the
two companies, The Information reports. The approach was part of an effort by OpenAI to persuade Amodei to replace Altman as CEO — but Amodei quickly turned
down the CEO offer.

—

November 19

Altman to meet at OpenAI HQ

According to The Information, Altman is expected to meet at OpenAI’s San Francisco headquarters as executives at OpenAI push to have him reinstated as CEO.
</item 3>

<item 4>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He said OpenAI still has a “deep bench of technical leaders” but its next executives will have to steer it through the challenges of scaling the business and
meeting the expectations of regulators and society.

Forrester analyst Rowan Curran speculated that Altman’s departure, “while sudden,” did not likely reflect deeper business problems.

“This seems to be a case of an executive transition that was about issues with the individual in question, and not with the underlying technology or business,”
Curran said.

Altman has a number of possible next steps. Even while running OpenAI, he placed large bets on several other ambitious projects.

Among them are Helion Energy, for developing fusion reactors that could produce prodigious amounts of energy from the hydrogen in seawater, and Retro
Biosciences, which aims to add 10 years to the human lifespan using biotechnology.
</item 4>

<item 5>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
one takeaway: go tell your friends how great you think they are. — Sam Altman (@sama) November 18, 2023

Investors pushing for Altman’s return

Investors — furious at the turn of events — are reportedly exerting pressure on OpenAI’s board to reinstate Altman, going so far as to recruit Microsoft.
Nadella is said to be sympathetic.

Board agrees to reverse course — in principle

The Verge reports that the board agreed in principle to resign and to allow Altman and Brockman to return. It waffled, however, missing a deadline yesterday by
which many OpenAI staffers were set to leave the company. Altman is said to be ambivalent about coming back and asking for “significant” governance changes.

—

November 17

Brockman demoted

Brockman says he got a text from Sutskever shortly after noon on Friday asking for a quick call.
</item 5>

<item 6>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
That’s according to Bloomberg, which in a brief — citing sources close to the matter — said that discussions are happening between Quora CEO Adam D’Angelo, one
current member of the OpenAI board, and Altman — and possibly other board members as well.

Board tensions boil over

The New York Times reports that, before his ousting, Sam Altman made a move to push out board member Helen Toner because he thought a paper she had co-written
was overly critical of OpenAI. That, among other issues, led to OpenAI’s current predicament. Speaking of, The Times indicates that negotiations to hire Altman
back continue — but that one major sticking point remaining is “guardrails” meant to improve Altman’s communication with the board.

—

November 20

Altman joins Microsoft

Sam Altman, Greg Brockman and colleagues announce that they’ll join Microsoft to lead a new AI research team.
</item 6>

<item 7>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
He was recruited in 2014 to take lead of the startup incubator YCombinator.

“Sam is one of the smartest people I know, and understands startups better than perhaps anyone I know, including myself,” read YCombinator co-founder Paul
Graham’s 2014 announcement that Altman would become its president. Graham said at the time that Altman was “one of those rare people who manage to be both
fearsomely effective and yet fundamentally benevolent.”

OpenAI started out as a nonprofit when it launched with financial backing from Tesla CEO Elon Musk and others. Its stated aims were to “advance digital
intelligence in the way that is most likely to benefit humanity as a whole, unconstrained by a need to generate financial return.”

That changed in 2018 when it incorporated a for-profit business Open AI LP, and shifted nearly all its staff into the business, not long after releasing its
first generation of the GPT large language model for mimicking human writing.
</item 7>

<item 8>
OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' (Fortune,
2023-11-18T15:33:09+00:00)
Altman also co-founded Worldcoin, a biometric and cryptocurrency project that’s been scanning people’s eyeballs with the goal of creating a vast digital
identity and financial network.

___

Associated Press business writers Haleluya Hadero in New York, Kelvin Chan in London and Michael Liedtke and David Hamilton in San Francisco contributed to this
report.
</item 8>

<item 9>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
After sending a Google Meet link, Brockman was told that he was being removed from the board as chairman “but was vital to the company and would retain his
role” as president, and that Altman had been fired.

Altman’s firing publicly announced

OpenAI published a post on its blog announcing the executive shake-up. The company’s management team was aware shortly after.

i loved my time at openai. it was transformative for me personally, and hopefully the world a little bit. most of all i loved working with such talented people.
will have more to say about what’s next later. 🫡 — Sam Altman (@sama) November 17, 2023

All-hands meeting

OpenAI held an all-hands meeting Friday afternoon during which Sutskever defended Altman’s ouster. He dismissed suggestions that pushing Altman out amounted to
a “hostile takeover,” and claimed that it was necessary to protect OpenAI’s mission of “making AI beneficial to humanity.”
</item 9>

<item 10>
A timeline of Sam Altman’s firing from OpenAI — and the fallout (TechCrunch, 2023-11-18T20:16:46+00:00)
We are collaborating to figure out the details. Thank you so much for your patience through this. — OpenAI (@OpenAI) November 22, 2023

Altman also posted about the new deal, giving some insight into the roller coaster that we’ve all been riding since his firing was revealed Friday. In his
words, his decision to join Microsoft on Sunday “was the best path for me and the team.” Since then, the new board’s composition and Microsoft’s support appear
to have been enough to bring him back to the AI company he co-founded.

Altman and board in talks

OpenAI’s board of directors is reportedly in talks with Sam Altman, ex-Y Combinator president and an OpenAI co-founder, to return to OpenAI as CEO as soon as
this week.
</item 10>
=== END ITEM id=7a648a62ad3e8581 ===

=== ITEM id=b4f017da6abc65fb ===
Gold sentences and facts:
Sentence 0: The process also loops in the European Commission to help facilitate dialogue, assess issues and bring pressure to bear on unfair practices.
  Fact 0.0: The process also involves the European Commission.
  Fact 0.1: The European Commission helps facilitate dialogue.
  Fact 0.2: The European Commission helps assess issues.
  Fact 0.3: The European Commission helps bring pressure to bear on unfair practices.
Sentence 1: We continue to work through the process with the European Commission and are focused on addressing its questions and any identified concerns at this stage.
  Fact 1.0: The speaker continues to work through the process with the European Commission.
  Fact 1.1: The speaker is focused on addressing the European Commission's questions.
  Fact 1.2: The speaker is focused on addressing any identified concerns at this stage.
Sentence 2: Its earlier actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.
  Fact 2.0: The earlier actions were focused on concerns about the spread of illegal content.
  Fact 2.1: The earlier actions were focused on concerns about disinformation.
  Fact 2.2: The illegal content and disinformation are related to the Israel-Hamas war.

Retrieved items:

=== END ITEM id=b4f017da6abc65fb ===

=== ITEM id=82bc1749e461cace ===
Gold sentences and facts:
Sentence 0: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 0.0: The full model, Gemini Ultra, was not released.
  Fact 0.1: Only a 'lite' version called Gemini Pro was released.
Sentence 1: Google’s public documentation says it uses three categories to rank results that are local to a searcher: relevance, distance, and prominence.
  Fact 1.0: Google's public documentation says Google uses three categories to rank results.
  Fact 1.1: The three categories are relevance, distance, and prominence.
  Fact 1.2: The categories rank results that are local to a searcher.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues these actions are anticompetitive means.

Retrieved items:

=== END ITEM id=82bc1749e461cace ===
