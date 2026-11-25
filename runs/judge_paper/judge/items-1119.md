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

This file has 1 items.

=== ITEM id=fc4947e3ba6b9e43 ===
Gold sentences and facts:
Sentence 0: Depending on which one you get, it might come with USB ports so you can charge multiple devices in one outlet.
  Fact 0.0: Depending on which one you get, it might come with USB ports.
  Fact 0.1: The USB ports let you charge multiple devices in one outlet.
Sentence 1: With a 10,000-mAh capacity, WIRED contributor Simon Hill says it charged his iPhone 14 Pro once and then delivered an additional 64 percent the second time before tapping out.
  Fact 1.0: The device has a 10,000-mAh capacity.
  Fact 1.1: WIRED contributor Simon Hill says the device charged his iPhone 14 Pro once.
  Fact 1.2: Simon Hill says the device then delivered an additional 64 percent to his iPhone 14 Pro the second time.
  Fact 1.3: Simon Hill says the device tapped out before the second charge was finished.

Retrieved items:
<item 1>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
In the consumer electronics space, we expect most companies to have some sort of Black Friday promotion, so that means you will likely be able to get big-ticket
items from brands like Apple, Samsung, Google, Sony, LG, Sonos and others for much cheaper prices.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.
</item 1>

<item 2>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
I’ve had my Slice Micro for a few months, so I’m not worried about it dulling yet, but it is unclear if the blade is replaceable.

Headphone hanger

Kaitlin Hatton, audience manager

Anchor Pro under-desk headphone hanger $ 12 $ 15 20 % off $ 12 $ 12 $ 15 20 % off An under-desk dual headphone hanger that uses 3M adhesive for mounting and an
included Velcro strap to anchor a headphone cable. $12 at Amazon

I gave this a try for one of our TikTok videos, and it has not disappointed me yet. I use it to hold my Razer Kraken headset and some extra cords. It’s small
enough to remain out of the way but large enough to hold more than the headset itself. It has a pretty strong hold and doesn’t give, even as I raise my standing
desk up and down several times a day.
</item 2>

<item 3>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
In terms of tablets, picking either of the two power bank options we recommend below should be a good fit for what you may have. For reference, all iPads
currently ship with 20 W power adapters, and so far, tablets from other manufacturers like Samsung and Amazon haven’t deviated far from that charging speed.

Whichever you choose, it never hurts to get some extra USB-C-to-USB-C cables. You can get this two-pack at Amazon for $15.99, so you can keep one at home and
then put the other in a bag you bring around during everyday travels.

Best power bank for your Switch

The Nintendo Switch console with detachable Joy-Cons can receive a maximum of 18 W through its USB-C charging port (the Switch Lite is slightly less demanding).
So, the very same compact power bank we recommend for phones and tablets can be used to charge your Switch.

Same note as before: We suggest buying more charging cables than you’ll immediately need (stow one permanently in the bag you frequently use).
</item 3>

<item 4>
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
Amazon’s fall Prime Day event is underway. For more deals we recommend across all categories, check out our full roundup here.

Despite what the name might suggest, Amazon’s fall Prime Day event isn’t just a day filled with deals for Prime members. Thanks to competing sales from
retailers like Best Buy and Walmart, everybody can take advantage of steep discounts without needing to pay a monthly premium to join in on the fun.

In some cases, retailers are matching Amazon’s pricing, meaning you can score the same discounts available to Prime members on items like the Kindle Paperwhite
and Echo Pop. There are also terrific deals available on other gadgets, including noise-canceling headphones, tablets, and smartwatches. To make the deal
hunting easier for you, we’ve rounded up all the discounted all of the worthwhile anti-Prime Day deals below.
</item 4>

<item 5>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
We think it's probably due for an update soon though.

Photograph: Amazon

If you like to take digital notes, the Kindle Scribe (8/10, WIRED Recommends) is the first of Amazon's e-readers that lets you write on the enormous 10.2-inch
screen like a regular notebook. However, if you want to write in the margins of books, you'll have to settle for using sticky notes. That's frustrating given
the point of spending this kind of money is to write on it (marking up books is better on the Kobo Elipsa).

Photograph: Amazon

The Fire Max 11 (5/10, WIRED Review) is Amazon's biggest, nicest, and most overpriced Fire tablet. This deal makes it much more palatable. The display and new
fingerprint sensor are nice, as is the ability to use the keyboard without resorting to Bluetooth, but the Fire OS operating system leaves much to be desired
(like a decent app store).
</item 5>

<item 6>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
It’s not the prettiest accessory one can attach to their desk, but it is highly functional.

Charging station

Jess Weatherbed, news writer

I got this as a birthday present from my partner after several months of arguing over our sleeping arrangements. Something about me trailing the charging cables
for my devices in the bed being “dangerous” and “extremely uncomfortable.” Anyway, after begrudgingly acknowledging my poor charging habits, I have to admit
that having this on my desk has had benefits outside of not garroting myself mid-slumber.

My iPhone, Apple Watch, and wireless earbuds rarely run out of juice, as I no longer fall asleep before plugging them in. I’ve also taken to using the charger’s
upright positioning for phones to my advantage — serving as a desk clock, a tiny display for Slack or Discord, and as a hub to remotely control the various
smart devices around my home. It helps me separate the device from being my phone and instead helps me build the habit of it being another tool to boost my
productivity.
</item 6>

<item 7>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
It’s all explained here.

Most Black Friday return policies extend through January 2024

According to Best Buy, certain products bought now (dating back to Oct. 27) until Dec. 30 may be returned before Jan. 13.

Target will accept most unopened items in new condition within 90 days of purchase.

Walmart has an extended holiday return policy in effect that allows items purchased between Oct. 1 and Dec. 31 to be returned on or before Jan. 31, 2024.

Amazon’s extended return policy is in effect for most items that are purchased between Nov. 1 and Dec. 31. Those items can be returned before Jan. 31, 2024.
However, Apple-brand products purchased in that time frame can only be returned on or before Jan. 15, 2024.

Black Friday likely won’t be your last opportunity to save

There will be amazing deals during Black Friday and Cyber Monday on everything, from tabletop and board games to toys, video games, and more.
</item 7>

<item 8>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Walmart will match its online Black Friday deals in stores starting November 10.

Will Amazon have a Black Friday sale in 2023?

Amazon will likely have a Black Friday sale in 2023. While the online retailer hasn’t made its deals known yet like Walmart has, Amazon will almost certainly
have deals across all of the types of products it sells, including electronics. We anticipate seeing record-low prices on Amazon devices in particular, like
Echo speakers, Fire TVs, Kindles and more. In years past, Amazon Black Friday deals have brought some of the best prices of the year on our favorite gadgets and
gear, including headphones, tablets, smartwatches, robot vacuums and more.

What other retailers have Black Friday sales in 2023?

Most retailers have Black Friday sales, so you can expect to see Target, Best Buy, and others join Walmart and Amazon in discounting inventory to capitalize on
holiday shopping interest.
</item 8>

<item 9>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
If you buy something from a Polygon link, Vox Media may earn a commission. See our ethics statement .

Your time — and money — is precious. Polygon’s What to Buy has rounded up the best in gaming hardware, entertainment, tabletop, merch, and toys, and how to get
them for the best value, so you spend less time shopping and more time enjoying.

A power bank, also commonly referred to as a portable charger, is an essential purchase for anyone who’s on the go with a phone, tablet, or a gaming handheld
like a Nintendo Switch or a Steam Deck. It lets you recharge your precious gadgets one or more times via a USB connection, so you can keep talking, texting, or
gaming throughout the day.

Power banks range in price depending on a few factors: how compact it is, what kind of ports it has, how much battery capacity it contains (typically measured
in mAh), and how quickly it can recharge your devices.
</item 9>

<item 10>
14 Best Deals: TVs, Keyboards, Mice, Chargers, and Earbuds (Wired, 2023-10-19T15:56:15+00:00)
Anker's tiny 511 30-watt charger is also excellent.

The Google Pixel Buds A-Series (8/10, WIRED Recommends) are our top overall earbuds pick. For nearly everyone–even some iPhone users–these check every box. They
get up to five hours of battery life (plus an extra 19 hours via recharging in their carrying case), IPX4 water resistance for workouts, and nearly instant
pairing with Android devices. This is a solid deal, but they do dip to $59 during major sale events, so it might be worth waiting for Black Friday.

Samsung Galaxy Buds2 Pro Photograph: Samsung

For Samsung owners, the Galaxy Buds2 Pro (9/10, WIRED Recommends) are one of our top earbud picks. They produce excellent sound quality, with a super
comfortable fit and IPX7 water resistance.
</item 10>

<item 11>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
The lengthy USB-C cable is capable of delivering 100 watts of power to beefy laptops like the 16-inch MacBook Pro, but you can also use it to power up
smartphones, tablets, and other gadgets.

for $7.99 ($12 off) at Amazon. The lengthy USB-C cable is capable of delivering 100 watts of power to beefy laptops like the 16-inch MacBook Pro, but you can
also use it to power up smartphones, tablets, and other gadgets. PopSockets is taking 25 percent off various grips, mounts, cases, and other accessories,
including this PopGrip for MagSafe phone handle / stand (on sale for $22.50) which is compatible with iPhone 12 phones and newer.

phone handle / stand (on sale for $22.50) which is compatible with iPhone 12 phones and newer. The worst thing about AirTags is you need something to stick them
in. Here’s a pack of four AirTag keychain holders for $8.91.
</item 11>

<item 12>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
With an IPX4 water-resistance rating, they'll also stay protected from sweat. They have excellent sound quality and offer an impressive nine hours of battery
life (with an extra 18 hours in the case). It's worth noting that many of the features won't work with an Android phone, but the core functions will.

Apple Charger Deals

Check out our Best Apple 3-in-1 Wireless Chargers and Best MagSafe Power Banks guides for more.

Anker 3-in-1 Wireless Charging Cube Photograph: Anker

This 2.5-inch cube from Anker is a great compact charger. It comes with a MagSafe pad on top (that charges at up to 15 watts), a top section that hinges to a
60-degree angle to reveal a charging surface for your AirPods, and a shelf on the side that has a built-in Apple Watch charger (which can comfortably
accommodate any Apple Watch including the Ultra). You'll also get a 5-foot cable and a 30-watt charger in the box. It supports fast charging too.
</item 12>

<item 13>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
The best deals of the year were once known to occur during Black Friday and Cyber Monday, which in 2023 take place on Friday, Nov. 24, and Monday, Nov. 27,
respectively. Not anymore!

In 2023, the best Black Friday deals kicked off on Nov. 17 at Amazon, Best Buy, Walmart, and other retailers. Instead of hosting all of the doorbuster deals on
one day right after Thanksgiving, retailers see the value in maximizing the amount of time you have to shop for products before Black Friday. And I’ve gotta
say, it’s quite nice to not feel rushed while shopping.

In addition to presenting you with the best board game deals and price drops on gaming monitors, plus deals on SSDs and microSD cards, there are some bits of
info and wisdom I’ve organized into sections below that might help you feel more informed as you begin to shop for the holidays.
</item 13>

<item 14>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
However, it likely won’t be the last opportunity for you to save. As someone who’s been covering deals for The Verge and Polygon for years, I’m shocked by how
often I see these low prices, often spoken of as if they’re a flash in the pan, extending through much of December. That said, it’s more likely that price cuts
on products you want can go by unnoticed unless you’re tracking deals yourself.

Track deals yourself

We love to tell you about great deals. But there’s a limit to the kind of products that we write about, and the ones that we see. If you’re not comfortable
leaving it up to the chance that we’re going to tell you about price cuts on the products you really want, you have the power to track deals with relative ease.

If you only shop on Amazon, CamelCamelCamel remains one of the best tools for tracking price changes and history for products.
</item 14>

<item 15>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
That's a $30 discount. Both the USB-C and Lightning models are on sale; the latter is compatible with Apple's iPhone 15 series as well as numerous Android
phones. We gave the Backbone a positive review last year, and we recommend it in our guide to the best gaming handhelds: It's more cramped than a traditional
gamepad, naturally, but it has all the requisite buttons to play modern games, it's extremely easy to set up and it works well for remote streaming from your
console or PC. If nothing else, it's a much better value than the PlayStation Portal.

PlayStation, Xbox and Switch video game sales

If you're all set on the hardware front, a truckload of PlayStation 5, Xbox Series X/S and Nintendo Switch games have dropped in price for Black Friday too.
Both Sony and Microsoft have rolled out holiday sales on their respective digital storefronts, while a number of physical games are down to new lows at Amazon,
Best Buy and others.
</item 15>

<item 16>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
It clicks right into the bottom of your device, and even comes in fun colors. This model has a 12-watt Lightning connector but there is a 22.5-watt USB-C
version for the iPhone 15 or Android phones for $22 ($10 off). Both connectors fold away when you aren't using them. If you use a particularly thick case, this
may not work. You'll likely be able to charge most phones fully once before needing to recharge the power bank itself via the included USB-C cable.

We are big fans of the Backbone One (8/10, WIRED recommends) and it appears in our Best Mobile Game Controllers guide. It slides open to cradle your phone and
is very responsive, with buttons and bumpers that feel nice and clicky. The USB-C version that is on sale will work with most Android phones and the new iPhone
15 range.

This 2-in-1 charger is featured in our guide to the Best MagSafe Accessories. The price matches the lowest we have tracked.
</item 16>

<item 17>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
If you want to stay powered up on your travels, you need a travel adapter, and this one from Epicka is the budget pick in our Best Travel Adapters guide. It
works in more than 150 countries and has four USB-A ports on the bottom, plus a 15-watt USB-C port on the side. It has dropped this low before, but not for a
while.

Get not one, not two, but four of our favorite smart bulbs for the best price we've seen all year. These smart bulbs are easy to use, beautifully vibrant, comes
with a variety of present colors, and work with Amazon Alexa and Google Assistant.

Photograph: Nanoleaf

We love this light kit for adding fun ambient lighting to any room. Each hexagon is its own light, so you can control and customize the hexagons into just about
any design of your choosing—both in color schemes. and onto your wall. Just grab a level to make sure you put them on straight!
</item 17>

<item 18>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
Whenever you come back to your workstation with your MacBook, all you have to do is plug it in and you instantly have a multi-monitor setup. With an output of
up to 60 watts, you can use the dock to charge your laptop too.

Claiming the top spot in our Best USB Flash Drives guide, the SanDisk Extreme Pro balances speed, reliability, and price. The sleek aluminum case has a loop for
attaching it to a keyring and a slider to push out the USB-A plug. It is fast, performs reliably (we've been using one regularly for two years), and comes with
a lifetime warranty.

Best Prime Day Tablet Deals

Photograph: OnePlus

The OnePlus Pad (8/10, WIRED Recommends) is one of the few Android tablets we think is a worthy iPad alternative. It offers good performance, has great battery
life, and an excellent 144-Hz, 11.6-inch LCD display.
</item 18>

<item 19>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is often a boon for deals on headphones and earbuds, and this year is no different. The shopping holiday has brought a plethora of deals on all
kinds of headphones, include several top picks from our guides to the best wireless headphones, wireless earbuds, budget wireless earbuds and gaming headsets.
Other popular models from the likes of Sony, Beats, Bose and Apple are discounted as well. All of this means it's an ideal time to pick up a new set, whether
it's to give as a gift to someone on your holiday shopping list or just to replace your own aging cans. To help you get a good bargain, we've rounded up the
best Black Friday deals on headphones and earbuds we could find for 2023.
</item 19>

<item 20>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
It was small and efficient, but it didn’t mesh well with the rest of my audio setup, especially on those sweltering summer days when I barely made it beyond the
confines of my own backyard.

A couple of years ago, however, I splurged on the Sonos Roam. The rugged, pint-sized device is on the pricier side when compared to other Bluetooth speakers,
but it produces solid sound for the size, offers wireless charging, and can automatically jump between my home Wi-Fi network and Bluetooth, a convenience I’ve
come to appreciate when strapping the speaker to my bike and heading out the door.

And while I might not be able to fire off my usual quips at Alexa when I take it into the backcountry — the Roam only supports voice commands when connected to
Wi-Fi — I certainly can still do it poolside with a drink in hand.
</item 20>

<item 21>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
Best portable charger for your phone and tablet

Today’s phones and tablets may differ in terms of their peak wired charging speeds, yet we’ve found some battery packs that should work well no matter what
device you have. It’s true that some outliers, like Samsung’s Galaxy S23 phone, can recharge at 25 W or higher depending on the model, but many other Android
phones and iPhones recharge via a wired connection at under 20 W — even less when you’re charging wirelessly.

How to pick the best power bank/portable charger for you Get one that lists the Power Delivery capability (noted as “PD” in the specs), and it should ideally
match, or come close to, the peak charging speeds (in watts) of the devices that you intend to charge with it. The batteries we recommend for phones and tablets
are also a good pick for the Nintendo Switch. However, the Steam Deck needs more juice, as do some mightier laptops that can receive power via USB-C.
</item 21>

<item 22>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
This helps support our journalism. Learn more.

Apple's Cyber Monday Promotion

Apple notoriously doesn't run sales on its own storefront. However, it does usually throw in a gift card for Apple Store credit if you purchase select items.
This year, now through November 27, you can get up to $200 in Apple Store credit to use later if you purchase certain items. For example, if you buy an iPhone
14 or iPhone SE, you can get up to $75 in store credit. Buy a MacBook Air and you'll get up to $200. You can find the full list here.

Separately, there's also a sale on Apple gift cards on Amazon. Each one has a unique corresponding coupon code that you can see on this page. For example, you
can get a $100 Apple Gift Card and a free $15 Amazon credit with code APPLET5USAP. There are tons of cards to choose from. You can browse them all here.
</item 22>

<item 23>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
That way, you can easily mount it on almost any tripod. Your phone (with the case protecting it) will slot easily into the SlimLink mount and stay secure.

What About iPhone Deals?

You typically won't see many iPhone deals. Those sales usually come from carriers like AT&T, Verizon, and T-Mobile, and have several requirements, like being on
an unlimited plan or opening a new cell line. If you're in the market for a new iPhone, we've laid out a few deals on older unlocked models here, along with
some carrier promotions.

Published November 26, 2023: We've updated prices throughout for Cyber Monday sales, and added a few more deals including the Otterbox sale, HyperDrive 6-in-1
Dongle, and Macbook Pro 16-inch laptop.
</item 23>

<item 24>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
Amazon does not price match, which is annoying, but it’s tolerable since it often has the lowest prices of all retailers anyway. Amazon only protects the price
on pre-orders that have its “Pre-order Price Guarantee,” which ensures that you’ll pay the lowest amount a product sells for before it’s launched.

Not only does Best Buy match prices from “qualified competitors,” but its site says that most products bought between now and Dec. 30 can be price-matched
through Jan. 13, 2024.

Walmart’s policy says it won’t price match with competitors, nor does it price match for products purchased online that later decrease in price.

Target says that purchases made between Oct. 22 and Dec. 24 can be price-adjusted if the cost lowers in the span of time, and if you show proof of purchase. You
may be eligible for a price match on a purchase, but the rules for submitting documentation vary depending on whether you’re shopping online or in store. It’s
all explained here.
</item 24>

<item 25>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
It also has a decent 1080p webcam. The chassis isn't especially light at 3.6 pounds, and we found the battery to last roughly seven hours in testing, which is
just OK. But for less than $400, this is a strong value for those who just want an affordable notebook for the basics.

FAQs

When is Black Friday 2023?

Black Friday 2023 lands on November 24 this year, with Cyber Monday 2023 following up a few days later on November 27. However, the early deals have already
begun. The entire month of November might as well be renamed Holiday Shopping Month as that’s when companies really start to get serious with their discounts.

What day is Black Friday 2023 at Walmart?

Walmart Black Friday deals will begin early this year on November 8. Walmart+ members will get a few hours of early access to Black Friday deals if they shop
online on November 8, with access for everyone else starting later that day.
</item 25>

<item 26>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
Nothing is more frustrating than buying a new pair of headphones, an OLED TV, or a backpack just to find out that you could have gotten it for a lot cheaper
somewhere else. In order to keep customers happy — and prevent them from going elsewhere — many retailers offer price-matching policies where they promise to
match a lower price you’ve found elsewhere. That’s information that could come in handy now that Black Friday and Cyber Monday are just around the corner.

What follows are the price-matching policies for a variety of major retailers. There are a few things to make note of here. First, all price-matching policies
have a number of rules attached to them. We’ve summarized some of the rules here, but you’d be advised to go to the links that we’ve provided and read the
policy carefully. Second, and logically enough, retailers that serve as marketplaces for other sellers, such as Costco, don’t have price-matching policies.

Lastly, there are always exceptions to the rules.
</item 26>

<item 27>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
That’s where my vacuum sealer comes in.

While I can’t speak to the quality of other vacuum sealers, the FoodSaver I have is awesome. Not only does the thing help keep raw meat and other food fresher
for longer in the fridge, but it also helps save space in the freezer (each package of meat becomes much flatter when all the air is sucked out of it). With
this little machine, I can load up whatever I want in one of the FoodSaver bags, insert the open end into the machine, which vacuums up all the air and then
closes the bag using its heat sealing feature in one fell swoop. It’s pretty neat!

For the traveler

Carry-on backpack

Kaitlin Hatton, audience manager

Earlier this year, I committed to traveling more, and so I took a look at the gear I had that could be replaced after years of trekking the globe. My ratty old
secondhand carry-on bag was the first thing to be replaced.
</item 27>

<item 28>
14 Best Deals: TVs, Keyboards, Mice, Chargers, and Earbuds (Wired, 2023-10-19T15:56:15+00:00)
It's light, comfortable for a wide variety of hand sizes and shapes, and features a couple of extra buttons without becoming cluttered. It is designed with
esports professionals in mind, meaning it has nice, clicky buttons, hyper-fast response time, and solid wireless connectivity even around 10 feet away from the
receiver.

This gaming mouse has a comfortable chassis; extra thumb buttons; and, perhaps most importantly, a modest price. Even when it's not on sale, it's one of our
more affordable picks, and this deal just makes it even more accessible.

Other Tech Deals

Anker Prime Photograph: Anker

Anker is kicking off its Black Friday sale early with discounts and a buy-one-get-another-50-percent-off promotion. Make sure you use the listed code to snag
the discounted price for each product. We generally like most Anker gear, but standouts include the Anker Prime power bank and the Anker Nano Power Bank.
Anker's tiny 511 30-watt charger is also excellent.
</item 28>

<item 29>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
One of the chargers we recommend below can match the speed delivered by Apple’s 140 W wall charger that’s included with 16-inch MacBook Pro laptops. On the
Windows side of things, most laptops don’t go above 90 or 100 W at peak USB-C recharging speed, so you should be set with our picks.

It might be tough to justify spending $100 to $150 on a power bank just to charge a laptop. I get it. However, it’s easier to justify getting one if you
frequently travel with multiple power-guzzling devices that you’d like to recharge simultaneously, as these are but a few available that can manage
multitasking.

Update (Nov. 17): Checked prices and stock for accuracy.
</item 29>

<item 30>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
Looking for a new Apple device? There's no time better than now. Whether you're on the hunt for a new iPad, MacBook, Apple Watch, or just a new iPhone case,
there are more Apple products on sale for Cyber Monday than you'll typically find at any other time of year. Here are some of our favorites. For more gadgets on
sale, be sure to check out our Best Cyber Monday Deals roundup.

We test products year-round and handpicked these deals. The discounts we show are based on actual street prices at retailers in the past few months. Products
that are sold out or no longer discounted as of publishing will be crossed out. We'll update this guide periodically.

Updated November 27: We've added a new deal on a 14-inch MacBook Pro (M3 Pro, 2023) and updated pricing and live deals as Cyber Monday comes to an end.

If you buy something using links in our stories, we may earn a commission. This helps support our journalism. Learn more.
</item 30>

<item 31>
The 26 best Amazon Prime Day deals you can still shop for today (Engadget, 2023-10-12T15:44:16+00:00)
OtterBox Fast Charge Power Bank

OtterBox Fast Charge Power Bank 15,000 mAh $36 $45 Save $9 See at Amazon

The OtterBox Fast Charge Power Bank is currently just $36 for the 15,000mAh capacity model, which is the version we tested and recommend in our portable battery
guide. This pack looks sharp and delivers a fast and reliable charge via USB-C or USB-A.

SanDisk 128GB microSDXC Card for Nintendo Switch

SanDisk 128GB microSDXC-Card Licensed for Nintendo-Switch $18 $31 Save $13 See at Amazon

If you need more room on your Switch, SanDisk's officially licensed microSDXC card is a good pick, particularly when it’s on sale. Right now a 128GB model is
within a dollar of its all-time low at just $18.
</item 31>

<item 32>
The 26 best Amazon Prime Day deals you can still shop for today (Engadget, 2023-10-12T15:44:16+00:00)
It's been an Engadget favorite for a while thanks to its sequential read speeds of 560MB/s and a standard 2.5-inch design that should make it easy to swap in
for your desktop's or laptop's used-up drive.

Other great deals for after Prime Day

Anker 511 Charger

Anker 511 Charger (Nano 3) $20 $23 Save $3 See at Amazon

New Apple Watches don’t come with power adapters, so if you’re looking for one that can deliver a quick charge, we recommend Anker's 511 (Nano 3) charging
brick. It’s down to $20 which is a mild, $4 discount, but still a welcome price for a worthwhile smartwatch accessory.
</item 32>

<item 33>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
It streams in 4K, supports HDR color, and is compatible with Alexa.

Games and gaming accessories

Mass Effect Legendary Edition $ 12 $ 60 80 % off $ 12 $ 12 $ 60 80 % off A compilation of the first three Mass Effect games repackaged and optimized for 4K with
reduced load times, improved performance, and various gameplay tweaks. $12 at Amazon (Steam code)

Samsung Evo Select microSD card (512GB) $ 25 $ 35 29 % off $ 25 $ 25 $ 35 29 % off This Samsung Evo Select microSD card offers 512GB of extra storage for a
Nintendo Switch, Steam Deck, or Asus ROG Ally at a good price point. $25 at Amazon

Staff favorites and miscellaneous tech

ChomChom pet hair remover $ 20 $ 32 38 % off $ 20 $ 20 $ 32 38 % off The ChomChom is a reusable cat and dog hair remover that works great for furniture.
</item 33>

<item 34>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
You can use it to charge your iPhone (at the fastest 15-watt charging speed Apple allows), fast-charge the Apple Watch Series 7/8/9/Ultra, and use the base to
power up your AirPods Pro (or AirPods with a wireless charging case) simultaneously all from a single outlet. It dipped as low as $100 back in August, but this
is still a great deal.

We like Anker's chargers and its Prime range employs GaN technology to shrink the size down. This 67-watt charger has fold-out prongs and offers two USB-C ports
and one USB-A. It supports fast charging for iPhones (in addition to Pixel phones and Samsung Galaxy phones) but can also charge a MacBook.

Photograph: Satechi

This handy charger has enough power to juice up two MacBook Pro models simultaneously. But it's also great if you have a lot of devices around your desk that
constantly need to be charged.
</item 34>

<item 35>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Black Friday and Cyber Monday may be in the past, but some sales are still hanging around (although maybe not for long). For more leftover deals we recommend
across all categories, be sure to check out the rest of the Cyber Monday still available here.

Every month or so, we like to ask our staff about their favorite stuff — whether it’s pet toys, travel aids, kitchen gadgets, or straightforward tech. And the
results are usually very different, very interesting, and a lot of fun.

In celebration of the annual post-Thanksgiving sales, we looked through some of our recent “favorites” articles and found deals on a lot of the tech, kitchen
tools, travel aids, and pet toys we like. We thought we’d list a few in case you’ve read about them in the past and thought, “Well, that sort of sounds good,
but it’s a bit pricey.”
</item 35>

<item 36>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
You can use it to charge your AirPods (as long as it comes with a wireless charging case) and other Qi-certified devices too.

Photograph: Zendure

The Zendure SuperTank Pro is the largest portable battery you can get away with taking onto an airplane. Which is why it's our favorite for air travel. With a
26,800-mAh capacity, four USB ports, and up to 100 watts in or out, it's the only portable power bank you'll need for every device on even extremely long
flights. Plus it kinda looks like a mini carry-on suitcase, that's neat. Just be sure to click the coupon button to get the full deal.

This is a great power bank if you don't want to carry a cable around. It comes complete with a built-in rotating USB-C or Lightning connector (MFi certified)
that slots into the bottom of your phone.
</item 36>

<item 37>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
One of the company's 67W USB-C chargers is 37 percent off, bringing it to $38 and matching its all-time low. Our top 65W fast wall charger, Anker's 735
GaNPrime, is down to a new low price after a 38 percent discount. And the Anker Nano Power Bank is seeing one of its first discounts since coming on the market
a couple months ago.

Meta Quest 2 VR headset

The Meta Quest 2 VR headset is $50 off and down to $249 at several retailers. That matches the best price we’ve seen. At GameStop and Target, you can get the
device with a $50 gift card. Despite the launch of the impressive Quest 3, we still consider the Quest 2 to be one of the best VR headsets available right now —
precisely because of its more affordable price.
</item 37>

<item 38>
Shop 200+ Walmart doorbuster deals for Black Friday (Mashable, 2023-11-25T01:13:44+00:00)
More TV deals

43 to 55 inches

65 inches

70 to 86 inches

Laptop deals

Why we like it

The total discount on this HP laptop isn't the biggest rollback we've seen so far, but it's a deceptively good deal on an underpriced laptop. With an Intel Core
i5 processor, this business 2-in-1 will be more than capable of handling your day-to-day work (and after-work streaming and surfing). While supplies last, you
can get this device for under $500.

More laptop deals

Traditional laptops

2-in-1 laptops

Gaming laptops

Headphone/earbud deals

Why we like it

If you're not ready to make the switch over to USB-C just yet, the second-gen AirPods Pro with a Lightning port are a stellar buy at only $169 — it's their
lowest price to date. (For its part, Amazon has never put them on sale for less than $189.)
</item 38>

<item 39>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is almost here, but we wouldn't blame you if you've felt like it's been here for a while. What was once a one-day event filled with limited-time
deals has grown into a month-long event with a plethora of sales to sift through each day. This year, Amazon, Target, Best Buy and Walmart and several other
retailers kicked off their Black Friday deals early, with some starting as soon as late October. If you're overwhelmed by the amount of discounts showing up in
your feeds, you're not alone.



To help, we at Engadget have collected all of the best Black Friday deals you can shop right now, primarily on tech we've tested and reviewed. These are the
best tech deals you can get this year, on the gadgets we know to be worth your money.
</item 39>

<item 40>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Here are the best Prime Day deals we could find gathered in one place.

Amazon's second Prime Day-like sale event for 2023 kicked off in the wee hours this morning, and it's turning out to be a great opportunity to get holiday
shopping done early. Or, if you missed out on any of the deals from Prime Day in July, this year's "Prime Big Deal Days" will let you make up for that. The sale
has brought back many of the same steep discounts we saw during the summer as well as new sales. We've also put together category roundups for deals on
headphones, tablets, TVs, video games and other gadgets.
</item 40>

<item 41>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
We believe the mid-tier plan, Surfshark One, will be the best for most people and a two-year plan is down to $2.69 per month, with an additional four months of
services thrown in for free.

Samsung M80C Smart Monitor

Samsung’s Smart Monitor M80C is $300 off and down to $400 right now. The selling point for this display, and really any Samsung Smart Monitor, is that it pulls
triple duty as an external display, a smart TV and an IoT hub. In addition to connecting it to your computer, you can use it standalone as a TV thanks to the
built-in TV operating system and included remote, and the embedded SmartThings hub lets you directly connect things like smart lights and other gear to the
display.

Motorola razr+

The Moto razr+ is 30 percent off and down to a record low of $700.
</item 41>

<item 42>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Older buildings especially suffer from what I can only describe as loose socket syndrome, where those very convenient modern boxy socket extenders with five
different USB and USB-C charging ports simply cannot stay in place and fall right out of the wall because they’re too heavy. After one (totally pleasant)
vacation where I had to charge my phone, watch, AirPods, and laptop in a weird corner of my room with the plug-in charging hub propped up on a strategically
balanced mountain of books and sham pillows, I bought this Anker combination extension cord / power strip. It’s not a full power strip — just a cube with a few
sockets along with USB and USB-C charging ports at the end of a five-foot cable. I’ve brought it on a few trips since then. It takes up extra space in my
suitcase but each time has left me feeling vindicated about the purchase.

The three prongs at the end of the cable are static, rather than folding flat for easy packing.
</item 42>

<item 43>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
A car-friendly charger

Sean Hollister, senior editor

Maybe someday Apple will realize that it wasn’t the brightest idea to artificially limit the reach of its MagSafe charging ecosystem and you’ll be able to plop
your phone on a magical minimalist disc that charges it at high speeds. In the meanwhile, a standard Qi charger is about the best you’ll get — and the bulky but
practical iOttie Easy One Touch Qi does it with the satisfying snap of springs. When you push your phone into its waiting jaws, it depresses a button that
causes those jaws to firmly snap closed on either side of your device. When you want to remove it, you pinch a pair of levers with your finger and thumb to
release as you grab your slab. It’s wide enough to fit practically anything on the market, save an opened Samsung Z Fold. I’ve used one for years with Android
and Apple phones alike, including newer MagSafe handsets.
</item 43>

<item 44>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
Aesthetics aside, it's also a fairly well-rounded case. The cutouts are all accurate, the buttons are great, and it comes with MagSafe support. We tested this
case with the iPhone 14, but this one is for the iPhone 15 Pro Max.

Photograph: Apple

Apple AirTags are our favorite Bluetooth trackers for iPhone users. This isn't a crazy sale, but getting these in a bundle of four means you can keep track of
every set of keys, your luggage for a trip, or even your kids' backpack all at once.

Those with an iPhone 14 or earlier can never have too many Lightning cables. These cables from Nomad are durable, available in different lengths, and come with
the choice between USB-C and USB-A for the other end.

Retailer Sale Pages
</item 44>

<item 45>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
We recommend Satechi's 108-watt wall charger if you own a MacBook Pro. It comes with three USB-C ports, each with the ability to charge at 100 watts if only one
device is connected (making it perfect for the 16-inch MacBook Pro). But if you want to charge three gadgets simultaneously, it'll charge them at 45, 30, and 30
watts, respectively, 58, 30, and 20 watts, or 65, 20, and 20 watts, depending on what you're charging. It's also on the larger side, but it's not as big as the
one from Apple, and the plug folds up too.

Satechi 165-Watt GaN Four-Port Charger Photograph: Satechi

This handy charger has enough power to juice up two MacBook Pro models simultaneously. But it's also great if you have a lot of devices around your desk that
constantly need to be charged.
</item 45>

<item 46>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
It fluctuates in price very often, but it's still a good deal.

Satechi's fold-up wireless charging stand has a 10,000-mAh capacity (complete with LEDs to show you how much juice you have left) and can wirelessly charge both
your phone and a wireless earbuds case (as long as it's Qi-certified). There's also a USB-C port to power up a third device. The only gripe we have is that it's
slow—delivering 10 watts of wireless charging power for phones (7.5 watts for iPhones), 5 watts for earbuds, and 10 watts via the USB-C port.

Apple 35-Watt Dual USB-C Port Compact Power Adapter Photograph: Apple

Apple's dual USB-C port power adapter is an essential accessory if you have a tendency to forget to pack chargers for your devices on the go. With two USB-C
ports, you can use it to recharge both your iPhone and MacBook simultaneously.
</item 46>

<item 47>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
It doesn’t have Matter or energy monitoring, but that’s alright at this price. $18 at Amazon (two-pack, with on-page coupon)

Amazon Smart Plug $ 15 $ 25 40 % off $ 15 $ 15 $ 25 40 % off Amazon’s apt-titled Smart Plug doesn’t require a hub and lets you add Alexa functionality to any
outlet, allowing you to control a range of devices with just your voice. $15 at Amazon$15 at Best Buy

Amazon Echo Glow $ 17 $ 30 43 % off $ 17 $ 17 $ 30 43 % off The Amazon Echo Glow is a multicolored smart lamp for children, one that can cycle through colors
and function as an Alexa-enabled sleep timer if needed. $17 at Amazon

TP-Link’s RE315 Wi-Fi extender is on sale for just $24.99 ($25 off), matching its best price to date.
</item 47>

<item 48>
The 26 best Amazon Prime Day deals you can still shop for today (Engadget, 2023-10-12T15:44:16+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Amazon's October Prime Day sale officially ended on Wednesday, but a few of the deals are still live as we head into the weekend. If you were hoping to check
off a few more names from your holiday gift list or buy something for yourself at a discount, see what remains on sale below. We combed through our coverage
over the past few days to find the best of what's left. While Amazon has returned most of their own devices, like Echos and Kindles, back to their full prices,
a handful of products from other brands like Google, Samsung and Anker, have maintained the lows they hit on Prime Day. Here are the best deals from Prime Big
Deal Days that you can still get right now.
</item 48>

<item 49>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
The performance is fast, the recharging is fast. It even looks fast. It has a bright, 120-Hz AMOLED screen, great speakers, and surprisingly good battery life
for all that. Our main gripe is the IP64 water- and dust-resistance rating, which is not nearly as good as other flagship phones.

Ugreen's charger is, as the name suggests, a 145-watt charger with a 25,000-mAh battery. It's surprisingly compact for the power it provides, although at 1.1
pounds, it's definitely not ultralight. There are two USB-C ports and one USB-A port. What sets the Ugreen apart is that you can actually draw 145 watts while
charging. That works out to one USB-C port at 100W and the other at 45W. Very few other batteries we've tested are capable of that feat.

Photograph: Anker

The Anker Nano is one of our favorite portable power banks, particularly for phones.
</item 49>

<item 50>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
The inside has some dividers but nothing too complicated.

But what I probably like most about this fanny pack (and this brand in general) is how colorful their bags are. I got one in lavender, stuck some enamel pins on
it, and now there’s absolutely no mistaking this is mine. I can see it easily from a distance, which also makes it harder to lose. The mix between colorful
whimsy and practical functionality is something I wish I saw more often. All I’m saying is that two of my friends went out and bought the same bag in different
colors as soon as they saw me wearing it. And we’re all immensely happy with our purchases.

Products for your pet

Hair remover

Mitchell Clark, former news writer

ChomChom pet hair remover $ 25 $ 32 22 % off $ 25 $ 25 $ 32 22 % off The ChomChom is a reusable cat and dog hair remover that works great for furniture.
</item 50>

<item 51>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It has a little stand to prop it up, a short cord that goes out to an outlet, and four USB-C ports that output 100 watts each. This is the lowest price we've
tracked.

Click the coupon button to see the discount at checkout. We recommend this Baseus charger if you're looking for a high-capacity option. With a 10,000-mAh
capacity, WIRED contributor Simon Hill says it charged his iPhone 14 Pro once and then delivered an additional 64 percent the second time before tapping out. It
comes with a USB-C cable too. The 6,000-mAh version is also on sale for $30 ($20 off). These deals are only applicable to Amazon Prime members.

Apple 35-Watt Dual USB-C Port Power Adapter Photograph: Apple

If you tend to forget to pack chargers for your devices, Apple's dual USB-C port power adapter is an essential accessory.
</item 51>

<item 52>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It can charge all your Apple devices at top speed and comes with a 6.6-foot cable and a 36-watt wall charger.

Apple hasn't nixed the charging adapter from its iPads yet, but it probably won't be long before you'll need to start buying your own. Or maybe you just want a
spare for another room. I like this tiny adapter from Anker. The 30 watts can fast-charge the iPad Pro despite the compact size and the plug folds in, making it
a nice option to take on the go.

What we like best about this phone mount is that you can use it one-handed. The arms automatically close around the phone when you place it against the trigger
button. To remove it, press the release bars. There's also a telescopic arm and ball joint that allows you to tweak the placement and set an ideal angle.
</item 52>

<item 53>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Backbone says the USB-C models will work with Apple’s new iPhone 15 lineup as well.

Crucial MX500

The Crucial MX500 internal drive is on sale for $46 right now. It's been an Engadget favorite for a while thanks to its sequential read speeds of 560MB/s and a
standard 2.5-inch design that should make it easy to swap in for your desktop's or laptop's used-up drive.

Your October Prime Day Shopping Guide: See all of our Prime Day coverage. Shop the best Prime Day deals on Yahoo Life. Follow Engadget for Prime Day tech deals.
Learn about Prime Day trends on In The Know. Hear from Autoblog’s experts on the best Fall Prime Day deals for your car, garage, and home, and find Prime Day
sales to shop on AOL, handpicked just for you.
</item 53>

<item 54>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
We have seen this dip a little lower, but this is still a solid deal.

The unusual design of Satechi's fold-up wireless charging stand allows it to charge both phones and a wireless earbuds case (if it supports Qi wireless
charging), plus a USB-C port to plug in a third device—not bad for a device that fits in your bag. It has a 10,000-mAh capacity with LEDs to show how much juice
is left. The downside is that it is slow, offering up to 10 watts of wireless charging power for phones (7.5 watts for iPhones), 5 watts for earbuds, and 10
watts from the USB-C port.

Best Prime Day Headphone Deals

Photograph: Sony

While its predecessor was one of the best pairs of wireless earbuds around, the WF-1000XM5 (7/10, WIRED Recommends) is still no slouch. They produce balanced
sound, are comfortable to wear, and have active noise cancellation.
</item 54>

<item 55>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Granted, it’s bulkier than buying one or two specialized adapter plugs, but if you’ve got a multi-continent itinerary, it’s a game-changer. What I like about
this one from Epicka is that it comes with a spare fuse in case things go sideways with voltage.

The only caveat is that, although it says “universal,” it’s technically only the four most common types of plugs. That’ll get you by in most countries, but it’s
not a guarantee in places like Brazil, South Africa, or India. Even so, I’ll take this over price gouging at airport electronics shops or having to take time
out of my schedule to visit a local hardware store.

Lovely lights

Brandon Widder, senior commerce editor

I’m a big fan of ambient lighting, even when I’m 50 miles from the nearest outlet.
</item 55>

<item 56>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
“Consumers are embracing mobile wallets to break down friction between discovering on social and purchasing on mobile.”

— Retailers are really pushing out discounts to get people more willing to spend money this year. Both Adobe and Salesforce said in the U.S. discounts were up
by about 28%. Categories that saw dramatic markdowns included toys, electronics and computers, per Adobe’s figures; respectively these saw sales up by 182% and
113% over last month.

— Buy Now Pay Later (BNPL) remains a popular option for paying upfront. BNPL drove $390 million in online spend, Adobe said, up 7.5% on last year.

We’ll update with more data later, and with Black Friday figures as they start to emerge.
</item 56>

<item 57>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

There are still tons of Apple Black Friday deals you can shop for online; these are our favorites.

Most of Apple’s current-generation devices went on sale for Black Friday — and most of those deals look like they’ll carry over through the weekend and into
Cyber Monday. That means if you were otherwise occupied on Friday, you can still save on iPads, Apple Watches, MacBooks, AirPods and more. Some of the prices
are new lows and many others match the record prices we’ve seen previously. We’ve tested and reviewed nearly every device below, so check out the links to our
full reviews and guides if you need help deciding which deals to snag. These are the best Apple Black Friday deals you can still get today from retailers like
Amazon, Target, Best Buy and B&H Photo.
</item 57>

<item 58>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
There's also a USB-C port to power up a third device. The only gripe we have is that it's slow—delivering 10 watts of wireless charging power for phones (7.5
watts for iPhones), 5 watts for earbuds, and 10 watts via the USB-C port.

This travel kit from ESR works well if you want something that can prop your iPhone in portrait or landscape orientation. It can also display your Apple Watch
in Nightstand mode, so you can peek and see how long before you must get out of bed. The main body folds open with a MagSafe charging pad for your iPhone and a
slot behind for your AirPods. Unfortunately, it doesn't support fast charging and it's capped at 7.5 watts and 5 watts for the iPhone and Apple Watch,
respectively. But it's a good, affordable charger. It dipped lower to $40 back in July, but this is still a solid deal.
</item 58>

<item 59>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It's not a huge discount, but this is the lowest price we've seen.

This charger packs two USB-C ports, one of which can output 65 watts when used alone. If you plug two devices in, the top will deliver 45 watts and the bottom
can do 20 watts. It also folds up and looks super sleek.

Satechi Duo Wireless Charger Stand and Power Bank Photograph: Satechi

Click the coupon button to see the full discount at checkout. The Satechi Duo combines the convenience of wireless charging with the portability of a power
bank. It has a 10,000-mAh capacity (complete with LEDs to show you how much juice you have left) and can wirelessly charge both your phone and a wireless
earbuds case (as long as it's Qi-certified). There's also a USB-C port to power up a third device.
</item 59>

<item 60>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Tile Slim (2022 version) $ 20 $ 35 43 % off $ 20 $ 20 $ 35 43 % off The Tile Slim is as thin as two credit cards, allowing you to stash the simple Bluetooth
tracker in your wallet, passport holder, or luggage. It also features a 250-foot range and, more importantly, compatibility with both Android and iOS devices.
$20 at Amazon$25 at Best Buy

Smart home tech

Meross Smart Wi-Fi Plug Mini (MSS110) $ 18 $ 23 22 % off $ 18 $ 18 $ 23 22 % off With solid performance and support for all of the major smart home platforms,
this affordable Wi-Fi smart plug gives you the basics you need to control non-smart devices for less. It doesn’t have Matter or energy monitoring, but that’s
alright at this price.
</item 60>

<item 61>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
There's even a built-in display to monitor stats like temperature and remaining battery percentage.

This heavy metal brick is a good option for charging laptops and smaller gadgets. It packs 20,000 mAh and supports a wide variety of fast charging standards.
There is one USB-C PD port rated at 65 watts, two USB-A QC ports at 30 watts apiece, and a micro-USB input (though you are best using the USB-C to recharge it).
We haven't seen it go on sale very often.

This travel kit from ESR works well if you want something that can prop your iPhone in portrait or landscape orientation. It can also display your Apple Watch
in Nightstand mode, so you can peek and see how long before you must get out of bed. The main body folds open with a MagSafe charging pad for your iPhone and a
slot behind for your AirPods. We have seen this dip a little lower, but this is still a solid deal.
</item 61>

<item 62>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
So when fellow video extraordinaire Vjeran Pavic (The Verge’s supervising producer) recently surprised me with a very cute birthday note and this magical little
gadget, I was elated.

And if that wasn’t enough, it has a carabiner hook. It is everything I have ever wanted in a rubber case and more. To have both my SD card reader and all my
cards in one place is priceless (well, actually $39.95). Thank you, Vjeran. <3

A mobile tripod for holiday snaps

Jess Weatherbed, news writer

When you go on holiday with your partner or family, you generally get stuck with three options when it comes to taking group photographs: a cramped selfie,
leaving someone out to take the picture, or asking a total stranger to take it for you. Not to be dramatic or anything, but I’d rather not ruin my vacation by
trying to figure out which individuals nearby are the least likely to dip the minute I hand over my phone.
</item 62>

<item 63>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
This 2.5-inch cube from Anker is a great compact charger. It comes with a MagSafe pad on top (that charges at up to 15 watts), a top section that hinges to a
60-degree angle to reveal a charging surface for your AirPods, and a shelf on the side that has a built-in Apple Watch charger (which can comfortably
accommodate any Apple Watch including the Ultra). You'll also get a 5-foot cable and a 30-watt charger in the box. It supports fast charging too.

Anker 737 Power Bank Photograph: Anker

We just added this power bank as our top upgrade pick in our guide to the Best Portable Chargers. It's pricey, but today's deal makes it more accessible. It
charges from zero to completely full in an hour and boasts a whopping 24,000-mAh capacity. And it's powerful enough to charge laptops and tablets as well as
phones and other gadgets. There's even a built-in display to monitor stats like temperature and remaining battery percentage.
</item 63>

<item 64>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
There's also a built-in fan that helps keep the temperature down when it's hot out.

Courant Catch:2 Photograph: Courant

This is a great wireless charger if you're looking for a stylish option to match your interior decor. The Belgian linen-wrapped surface comes in a variety of
muted tones including camel, charcoal, forest, and natural. It can charge two devices simultaneously, but you do have to be precise when placing your phone on
it. There are LED lights that show when it's charging and the rubber feet prevent the charging pad from shifting around.

If you need a charger for just your iPhone and AirPods, this is our favorite 2-in-1 stand. It has a MagSafe spot to charge your iPhone at top speed (up to 15
watts) and a spot below for your AirPods. It holds your iPhone in portrait or landscape orientation too, which is great for StandBy mode. It's not a huge
discount, but this is the lowest price we've seen.
</item 64>

<item 65>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
But that’s just life while traveling with a dog anyway.

An extension cord for awkward situations

Sarah Jeong, deputy features editor

Anker 321 Power Strip $ 15 $ 26 42 % off $ 15 $ 15 $ 26 42 % off This all-in-one 20W USB-C power cube boasts three AC outlets, two USB-A ports, and one USB-C
port. $15 at Amazon

Nobody wants to carry a power strip or an extension cord with them on their vacation. It’s probably unnecessary if you’re staying in relatively modern buildings
and definitely unnecessary if you’re camping. But sometimes you want to stay in a charming historical hotel or a lovely cabin in the woods, and it’s only when
you go to charge your devices at night that you realize that the only electrical socket in the bedroom is in the corner farthest away from the bed and there’s
already two lamps plugged into it.
</item 65>

<item 66>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
If you have an iPhone 14 or earlier, you can never own too many Lightning cables. These cables from Nomad are durable, come with the choice between USB-C and
USB-A for the other end, and are available in different lengths.

We're fans of Incipio cases—we've included them in our Best iPhones 15 Cases and Best Samsung Galaxy S23 Cases guides. For Cyber Monday, the company is offering
up to 50 percent off a wide selection of cases and accessories.

Photograph: Zugu

We named this the best iPad folio case we've ever tested. Why? It's durable, with a lip around the edges of the screen that protects the expensive glass if
dropped, and the buttons feel solid, yet easy to use. The magnetized cover stays securely shut, and it can be reversed to become a stand in eight different
angles. Don't have a 2022 10.9-inch iPad? No problem.
</item 66>

<item 67>
The best portable chargers for your gear (Polygon, 2023-11-17T13:10:12+00:00)
You can get this two-pack of 100 W-ready, six-foot-long, USB-C-to-C cables at Amazon for $15.99.

Related The best Nintendo Switch accessories

Best power bank for your Steam Deck

Each Steam Deck console includes a 45 W USB-C wall adapter, so it’s a good idea to find a power bank that can meet that power level when you’re on the go.
Thankfully, some options exist that can do this. One thing to note during your search is that only batteries that support the PD 3.0 spec, including the two
options below, can recharge the Steam Deck at its fastest speed.

Related The best Steam Deck accessories

Best portable charger for your laptop

The number of laptops that can recharge via USB-C is growing, making it possible to use the same charger to refill your laptop as you do your phone or Nintendo
Switch. For instance, Apple’s entire MacBook lineup allows it, as do several popular Windows laptops, like the Dell XPS 13, Razer’s Blade gaming laptop, and
more.
</item 67>

<item 68>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
Unfortunately, it maxes out at 7.5 watts for charging, but you can remove the USB-C charging cable when the iPhone is fully charged. There's also a built-in fan
to help keep the temperature down when it's hot out.

Anker Nano USB-C Power Bank Photograph: Anker

This is a great power bank if you don't want to worry about carrying a cable around, since it simply slots into the bottom of your phone—thanks to the built-in
rotating USB-C or Lightning connector (MFi certified). It also offers 5,000 mAh of energy, and there's a USB-C port for recharging it, complete with a short
cable. However, it might not be a good pick if you have a thick case.

This 2-in-1 charger is featured in our guide to the Best MagSafe Accessories. The price matches the lowest we have tracked. It doesn't have a built-in Apple
Watch charger, but it can top off your compatible wireless earbuds whilst simultaneously charging your iPhone.
</item 68>

<item 69>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
You can use it to charge your iPhone (at the fastest 15-watt charging speed Apple allows), fast-charge the Apple Watch Series 7/8/9/Ultra, and use the base to
charge your AirPods Pro (or AirPods with a wireless charging case) simultaneously all from a single outlet. It's normally around $150 but was discounted to $100
in August, so it's possible that Prime Day won't be your last opportunity to snag it for less.

Anker MagGo 637 Photograph: Anker

All you have to do is stick your iPhone to this adorable little orb and it'll start charging. On the back, you'll also find two USB-C ports, two USB-A ports,
and three AC outlets. It's both cute and tidy. It fluctuates in price very often, but it's still a good deal.
</item 69>

<item 70>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Every Black Friday and Cyber Monday, there’s always a ton of hype about the discounts on big-ticket items — you know, the deals on the OLED TVs, game consoles,
and tablets of the world. But it never fails that many of the ultra-cheap items make for some of the most clutch deals, namely because spending very little on a
small accessory you end up using all the time can be the most gratifying.

This year, throughout the annual shopping weekend, we’re once again keeping an eye out for the best tech deals on those little gadgets and gizmos under $25.
We’ve also got roundups for those working with $50 or $100 budgets, but the items below are pretty much as cheap as you can get in the tech world without simply
buying absolute crap that’s destined to end up as e-waste.

Some of these helpful budget deals include gems like fully compatible MagSafe chargers for the latest iPhones, a colorful smart speaker, Apple’s handy AirTag
trackers, and some staff favorites.
</item 70>

<item 71>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
(Or, “That’s pretty cheap, but maybe I’ll wait until the price goes down some more...”)

So here are some of our staff’s most-liked gear and gadgets, much of which is still discounted from Cyber Monday.

Tech tools

Electronics repair kit

Alex Cranz, managing editor

Tekton Everybit Tech Rescue Kit $ 22 $ 29 24 % off $ 22 $ 22 $ 29 24 % off A 46-piece screwdriver kit that can help you repair practically any tech device out
there. $22 at Amazon

I own at least two of these little Tekton Everybit Tech Rescue Kits, and I frequently buy them for friends and family, too. For an average price of $35, you get
a screwdriver with nearly every bit you’d need for most gadgets (including the weird ones for Apple products), a plastic and a metal spudger, tweezers, and a
suction cup. I’ve replaced batteries in iPhones with this kit.
</item 71>

<item 72>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
Flip over to the back, and you'll find two USB-A ports, two USB-C ports, and three AC outlets. It's a compact and tidy charger, and our unit has been going
strong for more than two years. It fluctuates in price very often, and it's hit as low as $60 in the past, but we still think this is a good deal.

This power bank is one of our favorites if you're looking for an upgrade. It has an absolutely gargantuan 24,000-mAh capacity, and it can be charged up in an
hour. It can output up to 140W, making it powerful enough to charge laptops and tablets, as well as charging your phone super quick.

This is a great charger if you travel often with an iPhone and Apple Watch. Instead of packing two separate chargers, the Duo offers the luxury of both all
within a compact, foldable form factor.
</item 72>

<item 73>
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
</item 73>

<item 74>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Come on in and peruse these affordable wares.

Phone accessories and smart trackers

Apple AirTag $ 24 $ 29 17 % off $ 24 $ 24 $ 29 17 % off Apple’s AirTag can help you find your lost items with its ultra wideband technology. You’ll get the best
compatibility with an iPhone, though Apple released an Android app that can detect an AirTag’s location and notify you if one seems to be following you. Read
our review. $24 at Amazon$80 at Amazon (four-pack)

Anker 511 Charger (30W) $ 16 $ 23 30 % off $ 16 $ 16 $ 23 30 % off Anker’s 30W 511 Charger juices phones quickly and is small enough to carry around in a purse.
</item 74>

<item 75>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Salesforce’s calculations of average order value were equally modest. Globally, average orders were up just 2% to $103 per “basket” while in the U.S. they were
up a paltry 1% to $119.

Mobile devices had a standout year: Adobe said that some $3.3 billion was spent over mobile devices on Thanksgiving, up 14% and an all-time record for the day.

Salesforce added that online traffic in general to e-commerce sites, which includes browsing, was up too, but again only in single digits of 4% globally and 6%
in the U.S.

The two are actively tracking sales for today, Black Friday, and for the whole of the “Cyber Weekend” — which not only kicks off holiday shopping but has a
particularly strong showing in online sales due to people traveling to be with family.
</item 75>

<item 76>
These Are Our Favorite Walmart Cyber Monday Deals (Wired, 2023-11-27T14:22:26+00:00)
Remember when people would break down the doors to get inside Walmart to score the best deals? Isn't civilization nice? Fill up your digital shopping cart
instead with these Walmart Cyber Monday deals on some of the best devices we've tested this year—no door-breaking necessary.

We test products year-round and handpicked these deals. The discounts we show are based on actual street prices at retailers in the past few months. Products
that are sold out or no longer discounted as of publishing will be crossed out. We'll update this guide periodically.

If you buy something using links in our stories, we may earn a commission. This helps support our journalism. Learn more.

Top Deals

Jump to a Topic: PC and Gaming Deals, Home and Kitchen Deals, TV Deals, Watch and Fitness Tracker Deals, Headphone and Earbud Deals, Smart Home Deals, Speaker
Deals

PC and Gaming Deals

Need a new laptop? We test dozens throughout the year.
</item 76>

<item 77>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
I’ve used one for years with Android and Apple phones alike, including newer MagSafe handsets.

A colorful fanny pack

Victoria Song, senior reviewer

I’ve always been stymied by the need for a bag smaller than a backpack or knapsack but larger than a dinky clutch. I was wary of the whole “wear a fanny pack as
a mini crossbody bag” trend, but earlier this summer, I caved and bought the Baboon to the Moon 3L Fannypack.

Now, I cannot go back. This bag easily fits my wallet, phone, house keys, car keys, hand sanitizer, and chapstick — everything I need when I take walks or run
errands. It’s also got a quick-release buckle, so I can stick a carabiner on it and schlepp around a water bottle on hot days. The material is also incredibly
durable, spill-proof, and easy to clean. The inside has some dividers but nothing too complicated.
</item 77>

<item 78>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
This is a great alternative — plus, you feel like kind of a badass when someone asks you to help fix a gadget, and you just pull this kit out of your bag in the
middle of Starbucks and get to work.

Safety cutter

Emilia David, reporter

I admit TikTok made me buy this small safety cutter, but it’s been indispensable to someone who may or may not have an online shopping addiction. The Slice
Micro Ceramic Blade safety cutter’s tiny blade cuts through paper packaging tape cleanly, opens plastic wrapping, and keeps me from going insane opening blister
packaging. It doesn’t damage whatever is inside, which unfortunately happens very often with my regular metal box cutter.

The downside is that it’s so small you may lose track of it if not in use, but it does have a built-in magnet and a handy dandy hole for a keyring. And while it
doesn’t fully slice through a cardboard box, it will still leave a scratch, although that could ultimately damage the ceramic blade if not used properly.
</item 78>

<item 79>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
It's our top pick in our best microSD card guide because it was the most consistent of the ones we tested, with some of the best overall speeds and all of the
important ratings that a good card should have: U3, V30 and A2. It'll be a great pick for your tablet, smartphone or Nintendo Switch, and it comes with a
full-sized adapter in case you need to use it with a camera or connect it to your computer.

Backbone One

The Backbone One is a well-built gamepad that snaps around the sides of your phone and works immediately with just about any game that supports controller
input. It’s also convenient for cloud streaming games from a console or PC. Right now both its USB-C and Lightning models are down to $70, which is $30 off.
That’s not an all-time low, but discounts on the device are relatively uncommon. We gave the One a positive review last year, and we currently highlight it in
our guide to the best handheld gaming systems.
</item 79>

<item 80>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Universal travel adapter

Victoria Song, senior reviewer

Epicka universal travel adapter $ 20 $ 25 20 % off $ 20 $ 20 $ 25 20 % off Epicka’s universal travel adapter is an all-in-one adapter that includes four
different plugs that cover over 150 countries. $20 at Amazon

In my youth, I forgot to pack plug adapters for international trips one too many times. Buying them once you’ve landed in another country isn’t always easy,
either. And if you’re like me, your relatives in rural Korea don’t always have more than one plug for your American devices — in which case, you’ll have to
share with your six other cousins. Nope. Absolutely not. Which is why I never leave this country without a universal travel adapter.

Basically, it’s six plug adaptors in one. Depending on which one you get, it might come with USB ports so you can charge multiple devices in one outlet.
</item 80>

<item 81>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
We found this pair to offer exceptional ANC, a comfortable fit and sound quality that has more bass plus "increased clarity and enhanced warmth" compared to
previous generations of the QC cans.

Anker Soundcore Space A40

The Anker Soundcore Space A40 is down to $54 for Black Friday, matching the lowest price we’ve seen. This is the top pick in our guide to the best budget
wireless earbuds: It’s comfortable, it supports wireless charging and multi-device pairing, it can last up to 10 hours a charge and its ANC is remarkably
effective for a sub-$100 pair. Its bassy sound should please most by default, but it can also be customized through a clean companion app. The only big
negatives are its middling call quality and lack of wear detection.

Dyson Black Friday deals

Dyson Black Friday deals include a $200 discount on the Dyson V15 Detect Absolute, bringing the cordless vacuum down to $550.
</item 81>

<item 82>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It dipped lower to $40 back in July, but this is still a solid deal.

Satechi's four-port hub is a little too large, considering the number of ports you get, but there's a good reason for that. Pop off the aluminum cover and
you'll find a slot for an M.2 SATA solid-state storage drive slot. That means access to an external SSD, along with two USB-A jacks, one 100-watt USB-C
pass-through charging port, and a 4K HDMI at the same time. I did not have an M.2 SATA drive to test, but this 1-terabyte card should work. Just know that
Satechi's hub does not support NVME M.2 cards.

Photograph: Otterbox

This is our Runner-Up pick in our guide to the Best Apple 3-in-1 Wireless Chargers.
</item 82>

<item 83>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
Amazon Prime Day Part II is here, and that means a fresh batch of Prime Day deals. Technically Amazon calls this Prime Big Deal Days, but like most people, we
think of it as Prime Day Deux. As usual, most of these Prime Day deals require a Prime membership, but you can snag a 30-day free trial to make the most of the
event. We've been combing Amazon's website to bring you the best discounts on laptops, tablets, kitchen and home gear, headphones, and plenty more.

We test products year-round and handpicked these deals. Products that are sold out or no longer discounted as of publishing will be crossed out. We'll update
this guide regularly throughout Prime Day by adding fresh deals and removing dead deals.

WIRED Featured Deals

Table of Contents

If you buy something using links in our stories, we may earn a commission. This helps support our journalism. Learn more.

Best Prime Day Amazon Device Deals

The discount will apply automatically during checkout once you meet the $40 order threshold on select products.
</item 83>

<item 84>
200+ of the best deals from Amazon's Cyber Monday sale (Mashable, 2023-11-27T08:45:59+00:00)
Table of Contents Table of Contents Echo, Fire TV, and Kindle deals Apple deals TV deals Laptop deals Headphone and earbud deals Tablet deals Gaming deals
Speaker deals Vacuum deals Kitchen deals Smart home deals Fitness deals Beauty tech deals Drone deals Camera deals Lego deals Gift card deals

UPDATE: Nov. 27, 2023, 5:00 a.m. EST This post has been updated with all of the latest Cyber Monday deals available at Amazon.

Amazon is dragging out the year's biggest shopping holiday(s) into 11 days of deals.

The retail giant began its Black Friday sale in the early morning of Friday, Nov. 17 (a week ahead of schedule) and was on top of making the switch to Cyber
Monday language in the wee hours of Saturday, Nov. 25. Official Cyber Monday mode, which is currently on through Monday, Nov. 27, includes both a ton of deals
carried over from Black Friday plus some new ones.
</item 84>

<item 85>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
The price matches the lowest we have tracked. It doesn't have a built-in Apple Watch charger, but it can top off your compatible wireless earbuds whilst
simultaneously charging your iPhone. It charges at the maximum 15-watt rate, and you can turn your iPhone sideways for iOS 17’s new StandBy mode, converting it
into a bedside alarm clock.

Anker 3-in-1 MagSafe Wireless Charging Dock Photograph: Anker

Anker makes our favorite lists often, including with this super compact 3-in-1 wireless charger. A MagSafe pad charges iPhones up to 15 watts at a slanted
angle, and on the side is a standard Apple Watch puck (no fast-charging support). In the space inside the triangle is where you can place your AirPods Pro (or
any other wireless earbuds case) to top them up.

This 2.5-inch cube from Anker is a great compact charger.
</item 85>

<item 86>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Today’s 5.5% rate is definitely an improvement on last year’s 2.9%. But it’s nothing compared to the years preceding COVID-19, such as 2017, where we were
seeing growth of 18% or more.

Inflation is making an impact but not as much as the worry around consumer spending, said Adobe, which said that spending is being led actually by more
discounts to encourage buying, rather than less buying of more expensive products. That leads one to wonder what kind of impact that’s having on retailers’
margins.

“Cyber Week is off to a strong start with Thanksgiving driving a record $5.6 billion in online spend as consumers took advantage of strong discounts and
continued their shopping plans, virtually,” said Vivek Pandya, lead analyst, Adobe Digital Insights, in a statement. “Mobile shopping hit an all-time high, as
shoppers took to their smartphones to get the best deals during holiday gatherings, further solidifying mobile’s growing importance in e-commerce.”
</item 86>

<item 87>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
This is key because the loose sockets of older buildings reject the beautiful convenience of folding prongs. There are more than enough sockets for one person,
and with some finagling (and maybe an extra charging brick), it can accommodate two people’s devices.

If you’re traveling overseas, don’t forget to purchase a different plug type for the region you’re going to or pack an adapter.

A portable smart speaker

Brandon Widder, senior commerce editor

Sonos Roam $ 134 $ 180 26 % off $ 134 $ 134 $ 180 26 % off The Sonos Roam is a truly portable Sonos speaker with a rugged design that’s built to withstand the
elements. It also features wireless charging and supports AirPlay 2, Alexa, and Google Assistant. $134 at Best Buy$134 at Sonos

For the longest time, my go-to portable speaker for camping and backpacking was the Ultimate Ears Roll 2.
</item 87>

<item 88>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
$21 at The Verge Store

Verge Deals / Sign up for Verge Deals to get deals on products we've tested sent to your inbox daily. Email (required) Sign up By submitting your email, you
agree to our Terms and Privacy Notice . This site is protected by reCAPTCHA and the Google Privacy Policy and Terms of Service apply.
</item 88>

<item 89>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
$16 at Amazon

Dbrand leather phone skins $ 24 $ 35 31 % off $ 24 $ 24 $ 35 31 % off Dbrand’s leather skins add a little grip to the back of a slick phone, but they’re mostly
a style choice. They’re high quality and non-damaging. $24 at Dbrand

EWA’s MagOne magnetic phone grip is on sale for $15.99 ($4 off) at Amazon. It’s a grip and kickstand in one, making bigger phones a little more wieldy without
getting in the way of MagSafe charging.

is on sale for $15.99 ($4 off) at Amazon. It’s a grip and kickstand in one, making bigger phones a little more wieldy without getting in the way of MagSafe
charging. You can buy Uni’s 10-foot USB-C to USB-C cable for $7.99 ($12 off) at Amazon.
</item 89>

<item 90>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
How to find the best Prime Big Deal Days deals?

WIRED is the only publication posting about Amazon Prime Day. Wait, sorry, looks a handful of other sites are, too. But you should only read WIRED, because we
actually vet every deal and compare it to the actual street price instead of an absurd MSRP that you will never observe in the wild. Also, we review the
products we recommend and don't just send you Prime Day deals on junk that will break. It's only a deal if you need it and it's good!

Are other retailers running sales?

Amazon Prime Day has inspired many imitators and you'll find sales from competitors like Best Buy, Target, and the like. Some of these sales are great, and
we'll flag when they are, but others are worth holding off on until Black Friday and Cyber Monday, when WIRED will again be the Internet's only website with
deals posts.

Retailer Sale Pages
</item 90>

<item 91>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget Editor Jon Turi found this pair to deliver more low-end depth than most of its peers, however. It can also last up to 10 hours per charge.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.
</item 91>

<item 92>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
And while I’ve long been a proponent of MPOWERD’s solar-powered Luci lanterns, I recently picked up the company’s like-minded string lights for car camping and
overnight jaunts in the backcountry when I don’t mind toting a little extra weight with me.

The 18-foot string is certainly not the brightest you can buy — it packs a series of 100-lumen LEDs, whereas your average headlamp might offer 400 — but it can
swap between six different colors and features a 2,000mAh battery for when your phone needs some emergency juice. Best of all, you can charge the lights via USB
or solar, meaning you can spend less time worrying about how to keep them going and more time taking in the vibes.

Memory card holder and reader

Becca Farsace, senior producer

As a video person who is constantly on the move, there is nothing better than the consolidation of gear — especially when it pertains to dongles.
</item 92>

<item 93>
41 Best Prime Day Apple Deals (2023) (Wired, 2023-10-10T22:58:00+00:00)
It charges at the maximum 15-watt rate, and you can turn your iPhone sideways for iOS 17’s new StandBy mode, converting it into a bedside alarm clock.

Anker 737 Power Bank Photograph: Anker

We just added this power bank as our top upgrade pick in our guide to the Best Portable Chargers. It's pricey, but today's deal makes it more accessible. It
charges from zero to completely full in an hour and boasts a whopping 24,000-mAh capacity. And it's powerful enough to charge laptops and tablets as well as
phones and other gadgets. There's even a built-in display to monitor stats like temperature and remaining battery percentage.

Nomad's wireless charger is one of our favorite MagSafe wireless chargers. It delivers the maximum 15-watt charging speed possible, but it's not meant to be
held up while you use the phone. It doesn't have a power adapter but includes a USB-C-to-USB-C cable. It'll sit securely on your desk and can charge an Apple
Watch, too.
</item 93>

<item 94>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
I’ve replaced batteries in iPhones with this kit. I’ve built entire PCs with this kit. I’ve swapped out backplates on Steam Decks and housings of Joy-Con
controllers with one of these kits. I’ve even used it to repair my eyeglasses.

One of the best parts of the kit is it all goes in a single case that can be tossed in a computer bag or purse or be left in a desk drawer at the office. But
honestly, the main reason I love it is the selection and quality of the bits. Too often, precision screwdrivers have super soft bits that strip the first time
you use them with a screw that’s been tightened by a machine. Given that most gadgets have at least one too-tight screw, I’ve gone through quite a few cheap
screwdriver kits before I settled on this one. While I’m slowly building out a high-quality selection of precision screwdrivers, most people don’t have that
luxury or necessity.
</item 94>

<item 95>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Ember Smart Mugs

Ember Smart Mugs are down to record-low prices for Black Friday — you can pick up a 10-ounce mug for as low as $90, or a 14-ounce cup for as low as $110. These
make great gifts for anyone on your list who may be otherwise hard to shop for, provided they enjoy a hot beverage every now and then. When used with their
companion app, these mugs can keep drinks at an optimal temperature for up to 80 minutes, or even longer if the cup is kept on its charging coaster.

TP-Link Kasa Smart Plug Mini

The smart plug we recommend for most homes is TP-Link's Kasa Smart Plug mini. A four-pack is on sale for $35 which is about $3 more than it sold for during
Amazon's October Prime day sale, :but still a decent $15 savings.
</item 95>

<item 96>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
Not Sonos fast there: Audio company Sonos scored a big $32.5 million win against Google a while back. Now it transpires that the company’s patents were deemed
unenforceable and invalid. Whoops. A federal judge threw out the $32.5 million win this week.

When it clicks, it really clicks: We took a deep look at Pixel 8, and our team discovered that it delivers solid performance and design upgrades. The camera got
a particularly enthusiastic shout-out, with Darrell declaring that Google’s Pixel 8 Pro camera is the new mobile photography champ, and Brian waxing lyrical
about the phone’s picture-snapper in “The camera’s still the thing.”

Who needs computers anyway?: It seems like all our mobile devices may be starting to cannibalize sales of personal computers — Ron reports that PC shipments
decline slows in Q3 2023 and that Apple’s sales plunge over 23%.

Let’s get together

Reddit’s API-powered chaos continues, but it appears that things are starting to resolve a little.
</item 96>

<item 97>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
If you're a frequent Target shopper, note that the retailer is selling this same bundle with a $75 store gift card for $450.

Xbox Series S

A starter bundle that includes a 512GB capacity Xbox Series S and a three-month subscription to Game Pass Ultimate is currently $240 at Microsoft. The console
by itself has a list price of $300 and three months of access to Xbox's service will set you back $51 after this year's price hike (though the first month is
often just $1 for new members).

Anker accessories sale

A slew of Anker chargers, cables, power adapters and other accessories are on sale for Black Friday at Amazon, with discounts ranging up to 37 percent. The
deals extend to our favorite premium power bank, Anker's Prime 20,000 mAH battery with charging base, which is $130 for the set and its lowest price ever.
</item 97>

<item 98>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
These have dipped as low as $159 in the past, but this is still a good discount.

AirPods Pro Photograph: Apple

Apple just launched these AirPods (9/10, WIRED Recommends) back in September, and they're already on sale. They're identical to the old pair on the outside but
pack some internal upgrades. In addition to slightly improved sound, there's also a new Conversation Awareness mode (that turns the music down and boosts
Transparency mode so you can hear someone talking to you).

Jump to a Topic: iPads, MacBooks, Apple Watches, Beats and AirPods, Chargers, Cases and Accessories.

Apple Charger Deals

For more options, check out our Best Apple 3-in-1 Wireless Chargers and Best MagSafe Power Banks guides.

Anker MagGo 637 Photograph: Anker

Simply stick your iPhone to this cute little orb and it'll start charging.
</item 98>

<item 99>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
You can just take it off to wirelessly charge your device.

This cable can do it all, which is why it's currently sitting at the top of our Best USB-C Cables guide. It can deliver 100 watts, transfer data at up to 40
Gbps, and serve an 8K display (or two 4K displays). It supports DisplayPort Alt Mode (DP Alt Mode), USB Power Delivery (USB-PD), and PPS, and it is
backward-compatible with all earlier standards. It's pricey, even with the discount, but it rarely dips this low. This is also the longest size; the shorter one
is also on sale.

The Creator Kit acts as the link between your phone and various mounting solutions. It comes with a magnetic locking mount that works with the Peak Design
SlimLink phone case and a few different mounting options including a Peak Design Arca quick-release plate, a GoPro mount adapter, and a 1/4-inch tripod mount.
</item 99>

<item 100>
54 Best Apple Black Friday Deals (2023): iPad, Apple Watch, AirPods (Wired, 2023-11-25T00:00:29+00:00)
It also packs a 5,000-mAh capacity and a USB-C port for recharging it (complete with a short cable). We don't recommend it if you have a thick case on your
phone, though.

The best 2-in-1 stand in our Best Apple 3-in-1 Wireless Chargers guide has a MagSafe spot to charge your iPhone at top speed (up to 15 watts) and a recessed
spot below for your AirPods case. It holds your iPhone in portrait or landscape orientation, which is great for StandBy. It's not a huge discount, but we've
never seen it cheaper.

Belkin 3-in-1 Wireless Charger Photograph: Belkin

This is our top charger in our Best Apple 3-in-1 Wireless Chargers guide. With a T-shaped design that suspends two of your devices midair, it's compact enough
that it won't take up too much space.
</item 100>
=== END ITEM id=fc4947e3ba6b9e43 ===
