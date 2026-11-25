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

=== ITEM id=c0545e81d5b4dcf6 ===
Gold sentences and facts:
Sentence 0: For now, Griffais explained, it's important to Valve that developers don't need to "split their attention" between different performance targets.
  Fact 0.0: Griffais explained that it is important to Valve that developers do not need to 'split their attention' between different performance targets.
  Fact 0.1: Griffais said this applies for now.
Sentence 1: The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day.
  Fact 1.0: The Steam Deck OLED will go on sale on November 16th at 1pm ET.
  Fact 1.1: Valve has Steam Deck OLED units ready to ship on that same day.

Retrieved items:
<item 1>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Manufacturers typically use die-shrinks to either improve performance or efficiency, and Valve has chosen the latter. The max frequency of both the GPU and CPU
are identical between the LCD and OLED models, so for developers tuning their games for Steam Decks, they only have a single performance target.

In concert with the more-efficient processor, there’s also a 25 percent larger battery with a 50Whr capacity, compared to the original’s 40Whr. According to
Valve, these changes combined result in a 30-50 percent increase in longevity between the two generations — the official battery life claim is "three to 12
hours." In our review, we measured a playtime increase of 62 percent when playing Hades on the OLED model versus our 18-month-old original. (Batteries degrade
over time, which explains why our gains were more pronounced.)

The new internals make the Steam Deck OLED marginally lighter, and also cooler to the touch when playing games, aided by a slightly larger fan.
</item 1>

<item 2>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
In early 2022, Valve released my favorite video game console. But even I had some gripes. With the Steam Deck, I could — for the first time — play the bulk of
my sprawling collection of video games wherever I’d like. But I tended to bundle it with a huge portable power supply. I also couldn’t quite shirk how much
better games looked on our family’s OLED TV.

And yet, for me, an adult with limited time to play video games alone in a quiet room, nothing could beat the Steam Deck’s freedom. Barring the occasional
diversion, Valve’s handheld PC had served as my preferred gaming option ever since I published our review. Then, last week, the Steam Deck OLED arrived. It’s as
if the Steam Deck team collected a laundry list of complaints about the original, then responded to all of them one by one.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences.
</item 2>

<item 3>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
In the past decade, Embracer grew its portfolio to cover more than 100 game studios, including Volition (Saints Row), Coffee Stain (Goat Simulator), Free
Radical Design (TimeSplitters) and Zen Studios (Pinball FX). The holding company also secured the rights to The Lord of the Rings in 2022, promising to turn it
into “one of the biggest gaming franchises in the world.”

Volition

In June 2023, Embracer announced a six-year, $2 billion funding deal had fallen through, and it was going to restructure — meaning, layoffs and studio closures.
Since this announcement, Embracer has shut down Volition, Free Radical Design and Campfire Cabal, it divested Goose Byte and it’s fired developers at Saber
Interactive. More than 900 people lost their jobs during these moves. Meanwhile, Embracer’s share price rose by 11 points in November.

This wasn’t the only layoff round of the year.
</item 3>

<item 4>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
The PDP Victrix BFG Wireless Controller for PlayStation 5, PS4, and PC in its Call of Duty: Modern Warfare II-branded black-and-gold design is on sale at Amazon
for $169.99 ($30 off). I know this controller looks very Extra (and it is), but it’s a very versatile DualSense alternative if you like FPS and fighting games.
It’s got multi-stage hair triggers, and its sticks and face buttons can be swapped out for an Xbox-style layout or as a six-button fightpad.

Verge Deals on X (formerly known as Twitter) / Join more than 51,000 followers and keep up with the best daily tech deals with @vergedeals Follow us!

Wireless gaming headsets, Laptops, and PC accessories

Asus ROG Flow Z13 $ 893 $ 1300 31 % off $ 893 $ 893 $ 1300 31 % off The ROG Flow Z13 is a gaming tablet with a detachable keyboard, kickstand, and plenty of
chops for QHD gaming.
</item 4>

<item 5>
Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more (Seeking Alpha, 2023-12-03T13:02:00+00:00)
(GTLB) is due to post its Q3 results after the market closes on Monday. With a 10% surge in the stock this year, Seeking Alpha's Quant Rating system suggests a
Hold, expressing concerns about valuation. This contrasts with Wall Street analysts, who advocate a Buy rating.

Recently, Raymond James has rated the stock with a Market Perform rating, citing potential market exuberance and growth potential in fiscal 2025 due to a large
total addressable market, healthy customers, and a value proposition favoring consolidation over fragmentation.

On the other hand, Cantor Fitzgerald started coverage with an Overweight rating and a $55 price target, as the company anticipates benefits from product
consolidation and the generative artificial intelligence wave.

Consensus EPS Estimates: -$0.01

Consensus Revenue Estimates: $141.04M

Earnings Insight: GitLab has beaten EPS and revenue expectations in 8 straight quarters.
</item 5>

<item 6>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Other improvements include support for both Wi-Fi 6E and Bluetooth 5.3, together with a new antenna array that will give a better experience when using both at
the same time. There’s also a reworked power supply (it has a logo now!) with a longer 2.5m cable that will charge the OLED model from 20 to 80 percent in “as
little as 45 minutes.” The thumbsticks have also been made 1mm taller, and the touch pads' haptics are slightly sharper.

Valve

The Steam Deck OLED starts at $549, and will replace all but the entry-level model in the range.
</item 6>

<item 7>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 7>

<item 8>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
“Not consistently candid” is a very diplomatic way of saying Altman lied.

It’s possible that Altman — and potentially OpenAI President Greg Brockman, who stepped down as chairman simultaneously, then resigned — wanted to make a bold
move that he knew the board would not like. It’s not uncommon for these deals to be hammered out quietly in smoke- (or vape-) filled rooms and then presented as
a fait accompli, but if it was controversial enough and the board found out about these maneuvers, it could be fuel for an ouster.

But what kind of deal would be big and dangerous enough for a summary dismissal of the CEO and famous face of the company? The man was onstage two weeks ago; I
just talked with him! What could have happened since then?

Few would be shocked if Microsoft, which is deeply, deeply embedded in OpenAI as an investor and customer, is a factor here. Could Altman have been working with
— or against — OpenAI’s patron in secret?
</item 8>

<item 9>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
It’s less top-heavy, and it does technically weigh less. These small adjustments have made a noticeable difference during my playtime, and the device feels
slightly more manageable as a handheld — or maybe I’m just too distracted by all of its bright, pretty colors to care about the cramping in my palms.

The thumbsticks on the new handheld have an improved texture that shouldn’t turn gray after a few months’ use, and they feel perfectly serviceable. In my
hands-on time so far, I've found that the OLED touchscreen responds immediately to input, and the haptic feedback on the built-in trackpads feels even more
precise, populating in reactive pin-pricks under the textured plastic.

I spent a few cozy nights on my couch swapping between the LCD and OLED Steam Decks, and honestly, after I’m done with this review, I’ll probably retire my
original model. The Steam Deck OLED is sharper and more responsive; indie games, AAA games and plain old Steam menus look so much better on the new hardware.
</item 9>

<item 10>
78 Absolute Best October Prime Day Deals (2023) (Wired, 2023-10-10T14:23:00+00:00)
We strongly recommend OnePlus’ magnetic keyboard, which is also on sale for $100 ($50 off), however, if you purchase directly from the company's website, you
can get the keyboard bundled for free.

The 9th-gen iPad (8/10, WIRED Recommends) is our favorite iPad for most people, even though it's one generation behind. Aside from costing less than the current
10th-gen model, it is still compatible with the same accessories as the first-gen Apple Pencil. It retains the physical Home button with Touch ID at the bottom
of the screen.

Pixel Tablet Photograph: Google

Google's Pixel Tablet (7/10, WIRED Recommends) is part tablet, part smart display. It comes with a charging dock that doubles as a speaker, so you can pump out
the tunes and get answers from Google Assistant at a moment's notice. Take it off the charger and you get a full Android tablet experience.
</item 10>

<item 11>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
They're both rated for water and dust resistance and you can pair two for stereo sound.

Roku Streaming Stick 4K

Our favorite streaming stick, the Roku Streaming Stick 4K, has dropped to $38 for this sale. It packs a ton of features into an affordable package, including 4K
HDR streaming capabilities, Dolby Vision, AirPlay 2 and private listening when you have a pair of headphones. And since it's so small, it makes a great travel
entertainment device, too.

LG B2

The LG B2 OLED TV is down to $847 for a 55-inch model. That’s a new low and roughly $150 off its average street price in recent months. The B2 is the step-down
model from the C2 — which is also on sale at $1,097 for a 55-inch set — as it’s not as bright or color-rich with HDR content, it runs on a slower processor and
it has two fewer HDMI 2.1 ports.
</item 11>

<item 12>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Yes. But you might find that the screen and the battery justify the double dip. For someone like me, who puts hundreds of hours into the Steam Deck each year,
the upgrade is inescapable.

And if you don’t have a Steam Deck? How many times can I say it: This is my favorite console ever. And it just got better.

The Steam Deck OLED will be available Nov. 16 to purchase through the Steam storefront. The hardware was reviewed with a unit provided by Valve. Vox Media has
affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links. You can find
additional information about Polygon’s ethics policy here .
</item 12>

<item 13>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The original Steam Deck had a 7-inch 800p LCD running at 60Hz. The new model squeezes a 7.4-inch OLED screen into the same space, significantly shrinking the
bezels around the display. The new panel runs at the same resolution, but with a refresh rate of up to 90Hz. It’s also HDR, covering 110 percent of the P3 color
space with a peak brightness of 1,000 nits — the old model maxed out at around 400 nits. Like all OLED screens, it has pure blacks and near-instantaneous
response times. As before, anti-glare etched glass is available, but only on the premium model.

The Steam Deck OLED has a die-shrunk version of the same custom APU found in the original Steam Deck. The new APU is produced on a 6nm process, rather than the
older 7nm process.
</item 13>

<item 14>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
If you want to jump in head-first with Proton services, the company has discounted Proton Unlimited, which includes access to VPN, Mail, Calendar, Drive and
Pass, to just under $9 per month for the first year.

LG A2 OLED TV

We expect to see many more TV deals as we get closer to Black Friday, but for now one standout is LG's 48-inch A2 OLED TV on sale for a new low of $550 at Best
Buy. This is an entry-level model from 2022, so it's limited to a 60Hz refresh rate, lacks HDMI 2.1 ports and doesn't support other gaming-friendly features
like VRR. That said, it's still an OLED TV, so it delivers deep contrast, bold colors, wide viewing angles and low input lag. At this price, it should be a nice
step-up option for smaller or secondary rooms. Just note that it can't get as bright as higher-end OLED sets, so it's best situated away from direct sunlight.
</item 14>

<item 15>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is the highlight of the device, while things like faster charging, improved antennas and smoother haptics are welcome bonuses. More
importantly, this is Valve renewing its commitment to portable PC gaming, and that’s a relief to see. The Steam Deck is the granddaddy of handheld PCs and Valve
has the resources to continue innovating in this space. For now, the OLED model is a half-step toward the Steam Deck 2, which may be the final version of the
device — unless Valve finally learns how to count to three.
</item 15>

<item 16>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The panel has a wider color gamut (110% DCI-P3, for the visual obsessives) and the beautiful, inky blacks that OLED dorks like myself adore.

Steam Deck OLED’s bigger, better battery Valve’s promise of 30-50% more battery raised one obvious question: Is that comparing the OLED model with a Steam Deck
running its original firmware (which had poor battery life), or a Steam Deck using the latest firmware? Valve’s answer: the latter! So if you’re a Steam Deck
owner, expect those big battery improvements over your current Steam Deck experience.

The OLED panel also has a 90 Hz refresh rate — perfect for frame doubling from 45 frames per second. In case that jargon means nothing to you, suffice it to say
that gameplay on the Steam Deck OLED looks smoother.
</item 16>

<item 17>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
It streams in 4K, supports HDR color, and is compatible with Alexa.

Games and gaming accessories

Mass Effect Legendary Edition $ 12 $ 60 80 % off $ 12 $ 12 $ 60 80 % off A compilation of the first three Mass Effect games repackaged and optimized for 4K with
reduced load times, improved performance, and various gameplay tweaks. $12 at Amazon (Steam code)

Samsung Evo Select microSD card (512GB) $ 25 $ 35 29 % off $ 25 $ 25 $ 35 29 % off This Samsung Evo Select microSD card offers 512GB of extra storage for a
Nintendo Switch, Steam Deck, or Asus ROG Ally at a good price point. $25 at Amazon

Staff favorites and miscellaneous tech

ChomChom pet hair remover $ 20 $ 32 38 % off $ 20 $ 20 $ 32 38 % off The ChomChom is a reusable cat and dog hair remover that works great for furniture.
</item 17>

<item 18>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 18>

<item 19>
Discord’s latest teen safety blitz starts with content filters and automated warnings (Engadget, 2023-10-19T15:00:31+00:00)
One recent update saw it improve app launch times on both platforms. In the near future, Discord says people can expect a new “Midnight” dark mode the company
claims reduces battery consumption on devices with OLED screens, more functional notification tabs and a new search feature for the settings menu to make it
easier to find the exact option you’re looking to tweak.

Last but not least, the company says it will begin rolling out Premium App Subscriptions to eligible developers in the UK and Europe, following a launch that
began in the US a few weeks ago. On the subject of third-party apps, bots and plugins, the company said it’s in the process of exploring how to make those
accessible across nearly every part of Discord. It provided few details on the effort, but said the goal is to allow people to access their favorite apps and
bots without those experiences being restricted to select servers.

“We're experimenting with a few different things, but the goal is for developers to reach more people with the awesome experience they're building.
</item 19>

<item 20>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
It’s always been in Google’s power to solve this problem. They make several billion dollars a year in unfairly earned profits from imposing this tax, which is
nothing compared to the money they make from search. For all the other benefits they get from Android, Google could solve this problem today if they wanted to.

Do you today believe that Activision Blizzard was really intending to build its own app store?

I don’t know about Supercell, but we know from the documents in the case that Riot was planning to distribute League of Legends directly through their website,
on mobile, exactly as they do on PC. And that’s what they planned to do until Google paid them off to not do that. Google’s payoff at the minimum dissuaded Riot
from distributing off Google Play.

I think they’re the strongest example of the three, but I’m curious about Activision.

Activision was building a store! We know they had a large team of dozens of engineers building a mobile app store to launch on Android.
</item 20>

<item 21>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
Other companies with unions established in the past two years include Avalanche Studios, Anemone Hug, CD Projekt RED, Experis Game Solutions, Keywords Studios,
Sega of America, Tender Claws and Workinman Interactive.
</item 21>

<item 22>
Govee’s colorful LED light strips are on sale just in time for Halloween (The Verge, 2023-10-28T16:40:43+00:00)
That’s not bad considering Google’s first-gen smartwatch retailed for $349.99 when it first launched a little over a year ago.

The first-gen Pixel Watch may not be as capable as the new Google Pixel Watch 2, but they have a number of strengths in common. Both offer health and
fitness-tracking features powered by Fitbit, allowing you to enjoy special features like FDA-cleared EKGs. The wearables also support Google services like
Google Assistant and Google Wallet. And now, thanks to the new Wear OS 4 update, you get access to the Google Calendar app, Gmail, and the new Safety Check
feature introduced with the new wearable.

There are some trade-offs, though. The Pixel Watch 2 offers superior battery life, whereas the last-gen model struggles to last a full day. The newer wearable
also sports some new tricks, including Sleep Profiles, abnormal heart rate tracking, and nightly SpO2 features.
</item 22>

<item 23>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The HDR OLED model is what they would have released the first time around, if the technology had existed before 2022. The handheld market has evolved quickly —
in just over a year, the Steam Deck proved there’s widespread demand for handheld PCs, and its success helped drive chip makers and display manufacturers to
build hardware specifically for high-fidelity mobile play. This is great news for Valve and the Steam Deck, and also the other handhelds coming out of companies
like Ayaneo, ASUS, Ayn, Logitech, Retroid and Razer. Hell, maybe even Playdate will get an HDR OLED display one day. (Just kidding; Playdate is literally
perfect).

So, no, the HDR OLED refresh isn't the Steam Deck 2, but it is a welcome improvement on an already good device. The Steam Deck OLED is the facetuned,
photoshopped, spit-shined version of Valve’s handheld, and it features tangible, clever improvements.
</item 23>

<item 24>
Discogs’ vibrant vinyl community is shattering (The Verge, 2023-10-07T14:00:00+00:00)
According to the sellers who spoke with The Verge, bugs abound: the European seller, for example, had just been dealing with an issue with the platform’s refund
button. “Discogs said it was PayPal’s fault, and PayPal said it was Discogs’ fault,” he says. “It caused stress for the buyers, and so I had to do direct
refunds — which meant I was refunding not just what I made but Discogs’ and PayPal’s commissions as well, effectively losing money on the refund.” Melkisethian,
speaking a month later, had just noticed a shift in the way shipments are processed that required manually entering information in steps that used to be
automated.

Besides the baseline functionality of the site, there are other improvements that could bring Discogs closer in line with its competitors. “There are other
seller tools and seller initiatives that we’ve been asking for for years that have never been done — like any kind of tie-in with Google, any kind of
integration with social media, the kinds of things basically other platform has,” says Martin.
</item 24>

<item 25>
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
$245 at Amazon$250 at Best Buy

Seagate Storage Expansion Card (1TB) $ 130 $ 220 41 % off $ 130 $ 130 $ 220 41 % off Proprietary SSD expansion for the Xbox Series X / S consoles. The
plug-and-play drives are designed to be as fast as the Xbox internal SSD and are sold in 512GB, 1TB, and 2TB configurations. $130 at Best Buy

The best TV and streaming device deals

2022 Samsung The Frame TV (55-inch) $ 1000 $ 1500 33 % off $ 1000 $ 1000 $ 1500 33 % off Samsung’s art-inspired 4K TV features a QLED display and an anti-glare
matte display, which provides artwork and photos with qualities that resemble an actual canvas.
</item 25>

<item 26>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Currently, more than 10,000 titles on Steam fall under the playable or verified labels, with more added every day.

When will the Steam Deck OLED be available?

The Steam Deck OLED will go on sale on November 16th at 1pm ET, and Valve has units ready to ship that same day. The company has lowered the price of its LCD
lineup and will sell through its backstock of 64GB and 512GB models until they’re all gone. It’ll continue selling the 256GB LCD model at a reduced price of
$400, while the 512GB OLED model will cost $550 and the 1TB OLED model will cost $650. There’s also a limited edition 1TB OLED version priced at $680 that comes
with a special carrying case, a translucent gray body and orange accents.

Valve

Wrap-up

According to the developers I spoke with at Valve, this is the definitive version of the Steam Deck.
</item 26>

<item 27>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
And critically, Microsoft has a habit of making its big blockbuster games available to play on Game Pass, for a monthly fee, on the first day they release.

Image source, EPA-EFE Image caption, Bethesda, which is owned by Microsoft, launched its new game Starfield in 2023 - but only on Xbox and PC

It's up to gamers to decide for themselves whether it's a better deal to buy the game or pay to access it.

But it is worth noting this concerned the UK regulator so much it made sure Activision Blizzard games couldn't be streamed exclusively on Xbox for 15 years.

That might be because Microsoft has form here - when the hotly anticipated Starfield came out in September 2023 to critical acclaim, it was available
exclusively on PC and Xbox, and playable on Game Pass from day one.

If games are still on PlayStation, what's in it for Microsoft?

Even with games still able to appear on PlayStation, it's Microsoft who will be winning out.
</item 27>

<item 28>
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
</item 28>

<item 29>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
The best deals of the year were once known to occur during Black Friday and Cyber Monday, which in 2023 take place on Friday, Nov. 24, and Monday, Nov. 27,
respectively. Not anymore!

In 2023, the best Black Friday deals kicked off on Nov. 17 at Amazon, Best Buy, Walmart, and other retailers. Instead of hosting all of the doorbuster deals on
one day right after Thanksgiving, retailers see the value in maximizing the amount of time you have to shop for products before Black Friday. And I’ve gotta
say, it’s quite nice to not feel rushed while shopping.

In addition to presenting you with the best board game deals and price drops on gaming monitors, plus deals on SSDs and microSD cards, there are some bits of
info and wisdom I’ve organized into sections below that might help you feel more informed as you begin to shop for the holidays.
</item 29>

<item 30>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Google Pixel Tablet

Photo by Cherlynn Low / Engadget Google Pixel Tablet $409 $499 Save $90 The 11-inch Pixel Tablet comes with a charging speaker dock that allows it to double as
a smart home hub. This deal marks a new all-time low. $409 at Amazon

The 128GB Google Pixel Tablet is down to $409, which is $90 off and a new low. The 256GB model is also on sale for $509. While we think Samsung’s Galaxy Tab S9
works better purely as a tablet, Google’s 11-inch slate still gets you a sharp LCD display, a speedy Tensor G2 chip and an easy-to-hold design that weighs just
over a pound. Its big hook, however, is that it can attach to an included charging speaker dock and double as a Nest Hub Max-style smart display. Android still
isn’t as optimized for large screens as iPadOS, but if that extra functionality sounds useful to you, the Pixel Tablet is worth a look.
</item 30>

<item 31>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
In October 2019, Epic internally said it might pursue an aggressive pursuit model with the Epic Games Store, where it would aggressively pay more for exclusive
games and really pull in more and more users. Are you in the aggressive pursuit model?

No, we’re in a different model, which is called the “Ridiculously aggressive pursuit model.” If it’s a 10-slide deck, that’s on slide 11.

Did you get a bingo on your Verge bingo card?

Yeah, I think it was almost a complete shutout. I think there was only one cell in the entire board that was missing. I was really impressed — the thing for
that bingo card that really impressed me was “lick the cookie.” Do you remember? You may not even be old enough, but “lick the cookie” was in the 1999 US v.
Microsoft antitrust trial.
</item 31>

<item 32>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
Surprise! Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it. It might not look like a big change from the outside, but the
Steam Deck OLED is a thorough mid-cycle refresh of a machine that’s not even two years old. At the same time, Valve is also adjusting the configuration of its
entry-level model, making it a far more compelling option. Apologies to anyone who literally bought a Steam Deck yesterday.

At an event held for press prior to the device's launch, Greg Coomer, a product designer at Valve, called the new model "the definitive version of the Steam
Deck." He explained that the device contains many things the company wanted to include at the LCD model's launch but wasn't able to, along with some additions
based on community feedback.

As its name suggests, the most noticeable upgrade to the Steam Deck OLED is the display. The original Steam Deck had a 7-inch 800p LCD running at 60Hz.
</item 32>

<item 33>
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
$50 at Best Buy

Google Nest Hub (second-gen) $ 60 $ 100 40 % off $ 60 $ 60 $ 100 40 % off The second-gen Nest Hub is Google’s smallest smart display, with a seven-inch
touchscreen that can automatically adjust itself to the lighting in your room. What’s new in the latest version is its ability to automatically track your sleep
patterns with its Soli radar chip. Read our review. $60 at Walmart

The best gaming deals

Sony InZone H9 $ 245 $ 300 18 % off $ 245 $ 245 $ 300 18 % off The H9 is at the top end of Sony’s InZone gaming headsets. It features active noise cancellation,
compatibility with both PCs and the PlayStation 5, and can simultaneously connect to two devices (one over 2.4GHz wireless and one over Bluetooth).
</item 33>

<item 34>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Samsung microSD card sale

If you need a new microSD card for your Nintendo Switch, Steam Deck or GoPro, a trio of Samsung microSD cards we recommend are also down their lowest prices to
date. The 128GB version of the Samsung Pro Plus is down to $11 at Amazon, B&H, Best Buy and others, while the 256GB and 512GB models are down to $18 and $32,
respectively. Beyond that, the 256GB Samsung Evo Select is on sale for a new low of $15 and the 256GB Samsung Pro Ultimate is available for $25.

The Pro Plus is the top overall pick in our microSD card buying guide, as it consistently ranked among the fastest cards we tested despite its relatively
affordable price tag. The Pro Ultimate offers faster read speeds, so it could be worth the step up if you often move files between your card and a computer.
</item 34>

<item 35>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
One of the company's 67W USB-C chargers is 37 percent off, bringing it to $38 and matching its all-time low. Our top 65W fast wall charger, Anker's 735
GaNPrime, is down to a new low price after a 38 percent discount. And the Anker Nano Power Bank is seeing one of its first discounts since coming on the market
a couple months ago.

Meta Quest 2 VR headset

The Meta Quest 2 VR headset is $50 off and down to $249 at several retailers. That matches the best price we’ve seen. At GameStop and Target, you can get the
device with a $50 gift card. Despite the launch of the impressive Quest 3, we still consider the Quest 2 to be one of the best VR headsets available right now —
precisely because of its more affordable price.
</item 35>

<item 36>
The Verge’s 2023 holiday gift guide for kids (The Verge, 2023-12-18T23:14:07+00:00)
Price: $119.99 Amazon Best Buy

8BitDo SN30 Pro (Special Edition) The 8BitDo SN30 is more than a small controller best suited for tiny hands. It also oozes nostalgic vibes to make you feel a
bit of what mom and dad did while playing the original SNES — albeit, with modern games on PC or the Nintendo Switch. What’s more, its see-through color schemes
are sure to entice just about everyone, regardless of what year you were born.

Price: $44.99+ Amazon Walmart 8BitDo

Nintendo Switch Lite (Isabelle’s Aloha Edition) The Steam Deck may have ushered in a new era of handheld gaming, but it’s a bit pricey and overbearing for a
child. That said, there’s still no substitute for the simplicity of Nintendo’s Switch Lite and its travel-friendly 5.5-inch display.
</item 36>

<item 37>
Teenage Engineering's K.O. II sampler proves the company can do cost-friendly cool (Engadget, 2023-12-19T15:00:38+00:00)
There's something of a theme running through Teenage Engineering's recent products. That theme is you need more money. The Field range represents the Swedish
company’s most exclusive music making gear. So when its website teased a new product with a colorful countdown, the wallets of Teenies everywhere braced for
impact. Once that timer hit all zeros, the big reveal turned out to be the EP-133 sampler. Or, to give it its full name, the EP-133 K.O. II 64MB Sampler
Composer. The real surprise though, was that it both looked cool and, at $299, was reasonably priced.

Fadergate

The countdown was really just the start. Barely 24 hours after the K.O. II was revealed units started landing in buyers’ hands. Within days, YouTube was awash
with first look videos and tutorials. Before our review unit even showed up, several users were complaining that theirs had defective faders.
</item 37>

<item 38>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 38>

<item 39>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
“A small subset of teams are good to go for the next 10 years, but others have these peaks and valleys, and we were somewhere in between,” Krankel said. “We
weren't in danger of anything going sideways. But we were at a spot where we're like, it would be cool to tether to somebody who has a similar vision, and
somebody that we could work with that would like, de-risk us.”

Netflix

This is the short-term benefit of being bought by a larger company, but there are downsides to relinquishing independence. Having a corporate overseer can
result in rigid production timelines, hindering a studio’s ability to pivot, and despite all of the promises otherwise, developers may be forced to adhere to a
specific tone, vibe or game-development structure. Owned studios are held accountable by people outside of the actual development of a game, and the bigger the
company, the further away its bosses are from the creative process.

The most extreme negative outcomes for an acquired indie studio are, of course, layoffs and closures.
</item 39>

<item 40>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
When should we expect a jury verdict?

Judge Donato told jurors that we’re looking at roughly five weeks of trial starting November 6th and extending through early December. The schedule suggests the
court will take Thanksgiving off, finish looking at all the evidence by December 4th, take a couple more days off before closing arguments, and then let jurors
come to a decision.

“I fully expect we’ll have plenty of time to reach a verdict well before the December holiday period,” said Judge Donato.

Okay, I’m sold — how can I watch the trial unfold?
</item 40>

<item 41>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This investment alone means any time you buy a game built on Epic’s Unreal Engine, Tencent (and Sony) is getting a cut. Tencent is the biggest investor in
games, with thousands of tendrils across the industry — if you played something this year, Tencent was probably involved.

On a smaller scale, companies like Netflix and Devolver Digital have also dabbled in acquisitions recently. Devolver started buying studios in 2020, and it now
owns Croteam, Dodge Roll, Doinksoft, Firefly Studios, Nerial and System Era Softworks. Annapurna Interactive bought South African studio 24 Bit Games in
November. Netflix launched its Games division in 2021, and it’s already purchased four studios, including Oxenfree developer Night School and Alphabear company
Spry Fox.

Night School co-founder Sean Krankel told Engadget in June that the move to Netflix was a boon for the studio, providing financial security, a dedicated working
space and plenty of marketing support for its projects.
</item 41>

<item 42>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
Email (required) Sign up By submitting your email, you agree to our Terms and Privacy Notice . This site is protected by reCAPTCHA and the Google Privacy Policy
and Terms of Service apply.
</item 42>

<item 43>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
It’s a lot to get through, as always — so we won’t delay. But first, a reminder to sign up here to receive WiR in your inbox every Saturday if you haven’t
already done so.

Most read

OpenAI throws a dev day: OpenAI hosted its first-ever developer conference on Monday, and the company had a lot to talk about. Some of the more notable items
announced were tools to create custom “GPTs” (i.e., domain-specific chatbots), new text-to-speech models, an API for the text-to-image model DALL-E 3, and an
improved version of OpenAI’s flagship model, GPT-4, called GPT-4 Turbo.

Mac attack: Brian reviewed Apple’s new 16-inch M3 Max MacBook Pro and the M3 iMac 24-inch.
</item 43>

<item 44>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
The immediate reality is a bit of a mess and a little complicated, but this, Valve says, will be the Steam Deck line-up moving forward:

256GB Steam Deck LCD ($399)

512GB Steam Deck OLED ($549)

1TB Steam Deck OLED ($649)

As a result of the changes, the 64GB LCD Steam Deck (previously sold at $399) will drop to $349, and the 512GB LCD (previously $649) will drop to $499. Once
Valve sells through its inventory of both, those lines will be discontinued. Valve says it will continue to support all LCD models with software updates.

So, the long-term lineup starts at the same $399 price, which gets you the old LCD Steam Deck, but with 256GB of storage instead of 64GB. The new step-up model
costs $20 more at $549, but comes with an OLED display and the improvements outlined above, plus double the storage.
</item 44>

<item 45>
Discord’s latest teen safety blitz starts with content filters and automated warnings (Engadget, 2023-10-19T15:00:31+00:00)
For users, we don’t want them to be gated from having these custom experiences at their fingertips,” Cherry Park, director of product marketing at Discord, told
Engadget. “In terms of the way we architecture and build it, there are a couple of solutions. Some are easier, some are more difficult, and you're going to see
us experiment with a few of them over the next few quarters.”

Discord promised to share more about its efforts around app portability in the near future. In the meantime, Nitro subscribers will get a chance to test new
features before they become available to the public.

Update, October 20, 2023, 11:20 AM ET: This story has been updated to note that these feature are rolling out in the coming weeks, not this week as the post
originally said. We also removed a mention that Discord would automatically delete potentially sensitive content from strangers. This was originally part of a
briefing we received on the new features, but it was not included in the final release.
</item 45>

<item 46>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
Because while Valve hasn’t upped the horsepower, it has improved practically everything else. The Steam Deck OLED has longer battery life, faster Wi-Fi, and a
stand-alone Bluetooth module. And the screen isn’t just OLED; it’s HDR OLED — the first HDR OLED screen on any gaming portable, with a peak brightness of 1,000
nits for HDR content and 600 nits for SDR (for comparison, the original screen’s peak brightness was 400 nits). And somehow, with all of these additions, the
Steam Deck OLED is lighter.

So, goodbye, old Steam Deck; hello, new Steam Deck.

The Steam Deck OLED delivers on Valve’s original promise

OLED is in the name for a reason.

The OLED panel in the new Steam Deck is lighter, thinner, and more power efficient than its LCD predecessor. It’s also bigger and faster. Though the device has
the same shape as the original Steam Deck, smaller screen bezels allow for a display that’s slightly larger, at 7.4 inches compared to 7 inches.
</item 46>

<item 47>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh
antitrust scrutiny as Google.

It’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in
the open, I should say — certainly they still wanted to keep them secret.

But I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they
were similarly self-serving in terms of deals.

I’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was
awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners,
you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent.
</item 47>

<item 48>
Teenage Engineering's K.O. II sampler proves the company can do cost-friendly cool (Engadget, 2023-12-19T15:00:38+00:00)
II. It’s still very portable, but feels a bit more “serious.” It’s also just very nice to look at, which is something Teenage Engineering is quite good at. The
model number, EP-133, indicates that we might see others in the line, so fingers crossed for giant calculator versions of other instruments, too.

In use

Fader fully checked and batteries in place, the K.O. II springs to life with a flourish of icons across its display. Those icons are actually fixed and not made
up of pixels. Teenage Engineering calls it a “Super segment hybrid display” which basically fuses the digital watch part with a bunch of colorful, cute custom
icons to let you know when certain modes or features are activated. It reminds me of the old Game & Watch handhelds where you can see where all the icons are
and they are simply switched on or off as needed.
</item 48>

<item 49>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
It sports drift-free Hall effect sticks and is compatible with PC, Nintendo Switch, and Android.

8BitDo Ultimate Bluetooth Controller $ 56 $ 70 20 % off $ 56 $ 56 $ 70 20 % off The 8BitDo Ultimate Bluetooth Controller comes with its own charging dock,
features remappable controls, and is compatible with the Nintendo Switch, Steam Deck, and Windows PCs thanks to its Bluetooth and 2.4GHz connectivity. Unlike
the 2.4GHz-only version, it features drift-free Hall effect sticks. $56 at Amazon (black)

The RGB-ified PowerA Spectra Infinity Enhanced Wired Controller for Xbox and PC is on sale at Amazon for $29.99 ($15 off). In addition to its colorful lights,
it’s one of the cheapest controllers around to sport hair trigger lockouts.

for Xbox and PC is on sale at Amazon for $29.99 ($15 off). In addition to its colorful lights, it’s one of the cheapest controllers around to sport hair trigger
lockouts.
</item 49>

<item 50>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
That said, its contrast, colors and motion performance will still be a significant upgrade if you’re upgrading from a more basic LED TV. LG’s new B3 TV promises
a cleaner webOS interface, but it’s priced a good way higher as of this writing. If you want an OLED TV for less than $1,000, the B2 should be a better value.

Hisense U6K

If you want a more affordable TV, the Hisense U6K is also on sale, with the 55- and 65-inch models down to $400 and $548, respectively. We’ve seen these deals
before, but each matches an all-time low. The U6K is has received generally positive reviews around the web, as it's one of the few budget-level sets with
quantum dots, full-array local dimming and a mini-LED backlight. All of that should help it deliver better contrast and color volume than most TVs in its price
range.
</item 50>

<item 51>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
These days, Steam generally peaks at over 30 million, with valleys of 20 million. Fortnite’s valleys are closer to 3.8 million.]

Tim Sweeney says he plays Jellie in Fortnite. Image: Epic Games

Okay, let’s do a lightning round. What’s your favorite Fortnite skin?

I play as Jellie! Jellyfish. You know, the big tentacles? It’s just so cool!

Will Epic ever announce Unreal Tournament 3 X?

Forever in motion, the future is.

Why is Fortnite still not playable on Steam Deck?

If we only had a few more programmers. It’s the Linux problem. I love the Steam Deck hardware. Valve has done an amazing job there; I wish they would get to
tens of millions of users, at which point it would actually make sense to support it.
</item 51>

<item 52>
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
</item 52>

<item 53>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
For developers, this means the device has the same performance targets as the original and there’s no need to create games specifically for the new hardware. In
the end, the device weighs about 30 grams less than the original model, or the equivalent of five quarters.

Engadget

In use

Which leads us to the bad news for my tiny hand gang. The Steam Deck OLED is the same monstrous size as the original: 11.7 inches long, 4.5 inches tall and
nearly 2 inches thick. The thumbsticks on the new model are actually 1mm taller than the first version. This was one of my main complaints about the Steam Deck
when I first reviewed it: I have smaller-than-average hands, and the Steam Deck looks and feels absolutely ridiculous when I’m holding it. It’s just massive, no
matter how long my manicure gets. However, the OLED model feels more balanced than the original. It’s less top-heavy, and it does technically weigh less.
</item 53>

<item 54>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hands on with the Pixel 8 Pro: Darrell took the newly unveiled Pixel 8 Pro for a whirl, and he liked what he saw. While very similar to last year’s model (the
Pixel 7 Pro), Darrell felt that the improved cameras, brighter screen and enhanced AI-powered features made it enough of an upgrade to (potentially) warrant a
purchase — minus the underutilized temperature sensor. Stay tuned for his full review.

Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board,
calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a
report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
</item 54>

<item 55>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
Meanwhile Sophie, who was named Ubisoft's content creator of the year, told the BBC she "just wants the games to win".

"We want everyone to be able to play video games that they want to play, we want them to be accessible for everyone," she said.

"Obviously, we're going to have our opinions, but if the end result is good games, that's fine with us."

This Instagram post cannot be displayed in your browser. Please enable Javascript or try a different browser. View original content on Instagram The BBC is not
responsible for the content of external sites. Skip instagram post by cadaea Allow Instagram content? This article contains content provided by Instagram. We
ask for your permission before anything is loaded, as they may be using cookies and other technologies. You may want to read Meta’s Instagram cookie policy,
external and privacy policy, external before accepting. To view this content choose ‘accept and continue’. Accept and continue The BBC is not responsible for
the content of external sites.
</item 55>

<item 56>
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
</item 56>

<item 57>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I think the philosophy change would do both of those companies much good.

If you get your way in terms of being able to freely put your own store on Android, do you believe that would only be a game store or would it also be an app
store? We look at Valve and we see a store that could be both, but they’ve decided to focus exclusively on games.

So the Epic Games Store isn’t a games store, right? It’s the store operated by Epic Games. So we have a lot of non-games there already. We have the Brave web
browser, we have a number of software creation tools including Unreal Engine, and there’s more coming, including some other awesome creation tools and
productivity tools. We’ll host any app anybody wants of any sort.

I think the gaming market is something we’re uniquely close to, and so I think we would likely be able to forge closer partnerships and opportunities in gaming,
but we’ll be open to everybody on Android as we are on PC.
</item 57>

<item 58>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
Similar to Walmart and Amazon, be aware that items at Target shipped or sold by third-party retailers can result in longer lead times. Product listings usually
spell out your options for pickup, same-day delivery, and standard shipping — complete with estimated availability times — before you add them to your cart.
</item 58>

<item 59>
Teenage Engineering's K.O. II sampler proves the company can do cost-friendly cool (Engadget, 2023-12-19T15:00:38+00:00)
I used extra caution, along with some needle-nosed tweezers to support the fader as I applied its cap and so far… so good?

Photo by James Trew / Engadget

The K.O. II name tells us that Teenage Engineering considers this something of a sequel to the original PO-33 K.O. Clearly the K.O. II isn’t a Pocket Operator,
but its retro desk calculator aesthetic does take subtle design cues from that series. At 12-inches diagonally, it’s in iPad territory size-wise. The K.O. II
also runs on AAA batteries (or USB power) which is another nod to the PO series. It’s hard to say how long it’ll run on those batteries and it’ll vary from
brand to brand, but I’ve been using some cheap rechargeables for over a week and they seem to be going strong.

Personally, I was never particularly enamored with the Pocket Operators and much prefer the form factor of the K.O. II.
</item 59>

<item 60>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
The GameSir G7 wired controller for Xbox and PC is on sale for $35.99 ($9 off) at Amazon. It’s a straightforward third-party gamepad, though it also has two
programmable rear buttons, clicky face buttons, and a paintable front shell that is attached with magnets.

for Xbox and PC is on sale for $35.99 ($9 off) at Amazon. It’s a straightforward third-party gamepad, though it also has two programmable rear buttons, clicky
face buttons, and a paintable front shell that is attached with magnets. Another nifty GameSir controller on offer is the T4 Kaleid, which is on sale for $33.59
($8 off). This is the first time the transparent wired controller with RGB lighting has gone on sale. It sports drift-free Hall effect sticks and is compatible
with PC, Nintendo Switch, and Android.
</item 60>

<item 61>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.
</item 61>

<item 62>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$893 at Amazon

Sony InZone H3 $ 58 $ 100 42 % off $ 58 $ 58 $ 100 42 % off The InZone H3 is the wired (3.5mm / USB-A) model in Sony’s latest lineup of gaming headsets, which
are compatible with both PlayStation and PC. You can customize the audio profile of InZone headsets with the InZone companion app and Sony’s Spatial Sound
Personalizer. $58 at Amazon$60 at Best Buy

Elgato Stream Deck Mini $ 59 $ 80 26 % off $ 59 $ 59 $ 80 26 % off Elgato’s Stream Deck Mini is a smaller, six-key version of its popular macro pads. Its LCD
keys can be customized to offer quick one-button macros during a livestream, or it can simply control shortcuts on your computer and connect to your smart home.
</item 62>

<item 63>
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
</item 63>

<item 64>
The Steam Deck OLED arrives November 16 with an improved screen and longer battery life (Engadget, 2023-11-09T18:00:32+00:00)
When asked about the potential for future hardware — like a return of the Steam Machine or an updated Steam Controller — there were no firm updates, but there
does seem to be something happening. Griffais agreed that enabling the Steam Deck to work as a hybrid living room console was a "side goal" of the team, and
noted that everything Valve does to improve SteamOS will benefit other devices. Coomer added that Valve is still working internally on "a bunch of controllers
and prototypes." If SteamOS is being used on larger displays, the team would want to augment that experience with better input options. "We don't have any
announcements on that front, but it's why we're building prototypes and thinking about the future there," Coomer said.

The Steam Deck OLED will be available to order on November 16th at 1PM ET. Valve expects units to begin shipping as soon as orders start to roll in. The updated
pricing on the existing LCD models is already in effect.
</item 64>

<item 65>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Even with improvements to the battery, thermals and weight distribution, the Steam Deck OLED is still more of a Wii U than a Vita. It’s a beast of a handheld,
and it makes the most sense as a portable console to use around the house, rather than a truly travel-ready device. The new model even comes with an extra-long,
2.5m charging cable, designed specifically with couch-based players in mind. I didn’t test out the docking station with the new Steam Deck, but Valve recently
released a SteamOS update that lets players take advantage of HDR and VRR on external displays, which only drives home the Wii U-ness of it all.

As far as software, you know the drill by now. Valve has rolled out labels for Steam games signifying how well they’ll work on the handheld, with four levels:
unknown, unsupported, playable and verified. Playable games generally come with a small caveat, like having to deploy the on-screen keyboard, while verified
games have been fully optimized for portable play.
</item 65>

<item 66>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The OLED screen makes for a fancier Steam Deck, but it’s the battery life (and some tweaks throughout the design to carve out more of it) that lets Valve’s new
handheld truly deliver on its promise of a portable gaming PC.

All the tiny, delightful improvements of Steam Deck OLED

Since the original Steam Deck launched, Valve has been improving the hardware through updates — its team estimates around 300 to date. That will continue with
both the Steam Deck OLED and the original Steam Deck, the latter of which will get a fresh BIOS update to align itself with some of the former’s upgrades. The
original device will see some battery life improvement; just don’t expect the OLED model’s jump of 30-50%.

Some changes — even smallish ones — demand new hardware. And so, alongside the big improvements, Valve has made a gaggle of minor physical upgrades with its new
iteration.
</item 66>

<item 67>
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
</item 67>

<item 68>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Valve Valve Steam Deck OLED SCORE 89 Engadget 89 Critics - Not yet scored N/A Users - Not yet scored N/A Pros Improved battery life

Bright and smooth visuals

Better thermals

Feels less top-heavy

Long charging cable Cons Still ridiculously large

Seriously it’s so big

Valve Valve Steam Deck OLED This isn't the Steam Deck 2 some are waiting for, it's just a welcome improvement on an already good device. $549 at Steam

OLED screen

The HDR OLED display looks fantastic. It features a wider P3 color gamut, pure blacks and a peak brightness of 1,000 nits, compared with the original max of 400
nits. In action, this means the screen’s bezels meld with the black of any game, creating a more polished frame for retro aspect ratios and providing room for
higher-contrast colors. The screen on the new model doesn’t support variable refresh rates (VRR), but it now tops out at 90Hz, while the original could only hit
60Hz.
</item 68>

<item 69>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The Steam Deck OLED now includes a Wi-Fi 6E radio (great if you have a modern router and connection to make use of it) and a dedicated Bluetooth module, which
allows for improved audio quality with wireless headphones, the option to wake the Steam Deck from a controller, and the capability to connect many Bluetooth
controllers for multiplayer. The haptics are noticeably improved, with a wider frequency range. And the touchscreen is markedly better — it’s responsive to the
point where I don’t feel the need to plug in a USB-C keyboard anytime I need to type a few words.

A review for spec specifics If you’re the sort of person who likes the nitty-gritty technical details of screens and batteries, we also recommend reading The
Verge’s review.

Valve also claims that the Steam Deck OLED is more repairable, which, mercifully, I didn’t have the need to test for myself. (Parts will once again be sold
through iFixit.)
</item 69>

<item 70>
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
</item 70>

<item 71>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I don’t know if that was on behalf of you and fellow developers. Can you tell us what that email was actually about?

Google didn’t leak the email; they leaked their ridiculously biased summary of it to Abner Li at 9to5Google. Don Harrison testified at trial that he didn’t
think Google leaked to the press. And then he was presented with a document including Google’s press team and Sameer Samat, head of all Android, basically
summarizing the articles that appeared as a result of Google’s leaking of our plans to the press.

So I was wondering if you could tell me what the email actually said.

Oh, it’s in evidence! This was the one I mentioned! Right before Christmas-ish 2019, we were planning to launch the new Marvel season with a bunch of Star Wars
content and its awesome lineup. I asked Google’s executives to let Fortnite come to the Google Play Store using our own payment method and to open up, you know,
using their own payment methods to all developers, not just Epic.
</item 71>

<item 72>
Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more (Seeking Alpha, 2023-12-03T13:02:00+00:00)
Also reporting: MongoDB (MDB), SentinelOne (S), Asana (ASAN), The J. M. Smucker Company (SJM), AutoZone (AZO), Stitch Fix (SFIX), Box (BOX), AeroVironment
(AVAV), Yext (YEXT), Signet Jewelers Limited (SIG), Dave & Buster's Entertainment (PLAY), Ocuphire Pharma (OCUP), Hovnanian Enterprises (HOV), Designer Brands
(DBI), HealthEquity (HQY), Ferguson (FERG), G-III Apparel Group (GIII), The Descartes Systems Group (DSGX), Phreesia (PHR), Core & Main (CNM), J.Jill (JILL),
Daktronics (DAKT), Lands' End (LE), and more.

Wednesday, December 6

GameStop (GME)

Video game retailer GameStop (GME) is scheduled to post its Q3 earnings after the closing bell on Wednesday.
</item 72>

<item 73>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 73>

<item 74>
Save on the winners from the 2023 Game Awards this weekend (Polygon, 2023-12-09T15:00:00+00:00)
We’re coming up for air from our coverage of The Game Awards to loop our readers in on some of the best deals happening this weekend. If you’re looking to
experience some of the winners from the 2023 Game Awards for yourself, you can find discounts on a variety of titles on Steam. Some other highlights you might
want to check out include a sitewide sale on Blu-Rays from The Criterion Collection, and the Xbox Series X matching its Black Friday price on Walmart and
Amazon.

As we do every weekend here at Polygon, we’ve rounded up a collection of our favorite deals from the worlds of gaming and entertainment, along with some of the
best-selling products that have made a recent appearance on our site.

The best gaming deals this week

Steam is discounting many contenders from the 2023 Game Awards and winners from the past nine years.
</item 74>

<item 75>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
It's our top pick in our best microSD card guide because it was the most consistent of the ones we tested, with some of the best overall speeds and all of the
important ratings that a good card should have: U3, V30 and A2. It'll be a great pick for your tablet, smartphone or Nintendo Switch, and it comes with a
full-sized adapter in case you need to use it with a camera or connect it to your computer.

Backbone One

The Backbone One is a well-built gamepad that snaps around the sides of your phone and works immediately with just about any game that supports controller
input. It’s also convenient for cloud streaming games from a console or PC. Right now both its USB-C and Lightning models are down to $70, which is $30 off.
That’s not an all-time low, but discounts on the device are relatively uncommon. We gave the One a positive review last year, and we currently highlight it in
our guide to the best handheld gaming systems.
</item 75>

<item 76>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI reportedly delays the launch of its GPT store to 2024

According to a memo seen by Axios, OpenAI plans to delay the launch of its highly anticipated GPT store to early 2024. Custom GPTs and the accompanying store
was a major announcement at OpenAI’s DevDay conference, with the store expected to open last month.

November 2023

ChatGPTs mobile apps top 110M installs and nearly $30M in revenue

After launching for iOS and Androidin May and July, ChatGPT’s have topped 110 million combined installs and have reached nearly $30 million in consumer
spending, according to a market analysis by data.ai.

ChatGPT celebrates one-year anniversary

OpenAI hit a major milestone: one year of ChatGPT. What began as a “low-key research preview” evolved into a powerhouse that changed the AI industry forever.
</item 76>

<item 77>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
(Parts will once again be sold through iFixit.) However, I did enjoy the new power supply, with its faster charging (45 minutes gets you from 20% to 80%) and
2.5-meter cable that stretches the length of my couch.

One small, charming detail: The carrying case now has a tinier carrying case within it. If you want to be extra cautious, stick to the original bulky case. But
if you need to toss the Steam Deck in your backpack for a day trip? Go with the slim case that has funny little thumbstick nipples!

All of the tweaks, large and small, amount to a perfect handheld — made better by the price, which remains the same as the original Steam Deck available today,
just that the OLED will come with more stuff.
</item 77>

<item 78>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
[Editor’s note: Every journalist in the courtroom agreed the jury was attentive; most were also surprised by the rapid verdict.]

They got it, they got it quickly, and they were able to pull apart what was going on and contrast it with the fiction of the story that Google was trying to
tell.

What was something that surprised you in the courtroom, something you heard come to light that you hadn’t heard before?

I’d suspected a lot of the practices that Google had, you know, since 2018 or so when we first started this, to such an extent that some folks would
occasionally call me a conspiracy theorist. It was really, really interesting to see that my understandings of what Google was doing behind the scenes were
actually true — you’re leaking our conversations to reporters to get negative stories written about us; you’re paying other developers off to convince them not
to launch their own stores; they were going around and paying carriers and OEMs secretly not to carry competing stores.
</item 78>

<item 79>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is almost here, but we wouldn't blame you if you've felt like it's been here for a while. What was once a one-day event filled with limited-time
deals has grown into a month-long event with a plethora of sales to sift through each day. This year, Amazon, Target, Best Buy and Walmart and several other
retailers kicked off their Black Friday deals early, with some starting as soon as late October. If you're overwhelmed by the amount of discounts showing up in
your feeds, you're not alone.



To help, we at Engadget have collected all of the best Black Friday deals you can shop right now, primarily on tech we've tested and reviewed. These are the
best tech deals you can get this year, on the gadgets we know to be worth your money.
</item 79>

<item 80>
Teenage Engineering's K.O. II sampler proves the company can do cost-friendly cool (Engadget, 2023-12-19T15:00:38+00:00)
Before our review unit even showed up, several users were complaining that theirs had defective faders. Enough folks were having this problem that it quickly
became known as “fadergate.” One brave creator even took their unit apart and, possibly, discovered the cause — the internal pins were bent and not making a
connection. I asked Teenage Engineering about the issue and will update this story once I hear back.

Some buyer’s theorized that the issue might be caused by the fact that the K.O. II ships without the caps on its rotaries and fader. They come loose in the box
to enable the packaging to be flatter, but the cap for the fader is unusually tight fitting. This led to speculation that the enthusiasm required to push this
cap down might be putting too much pressure on components inside, opening up the cruel possibility of users breaking their own devices before they even got to
play with them.
</item 80>

<item 81>
Apple 16-inch M3 Max MacBook Pro review: A desktop among laptops (TechCrunch, 2023-11-06T14:00:15+00:00)
The triangle computer shader makes processing complex geometrical images more efficient. That, in turn, allows for higher-quality graphics and more scene
detail.

The results are impressive. I’ve been spending a bit of time with the Soulslike title, Lies of P. It’s basically Pinocchio with more swords. It’s of a small –
but increasing – number of day-and-date releases for macOS. A couple of years ago, it would have been inconceivable for the operating system to get the game the
same day as PS5, Xbox Series X/S and Windows. That said, the macOS gaming library still has a long way to go.

Indeed, the gameplay was smooth and detailed – something that also would have seemed impossible not all that long ago. Sony’s PS5 DualShock controller connects
seamlessly and makes for a fun portable gaming experience.

One thing I noticed during last week’s presentation is that Apple is leaning less heavily on Intel comparisons.
</item 81>

<item 82>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Here are the best Prime Day deals we could find gathered in one place.

Amazon's second Prime Day-like sale event for 2023 kicked off in the wee hours this morning, and it's turning out to be a great opportunity to get holiday
shopping done early. Or, if you missed out on any of the deals from Prime Day in July, this year's "Prime Big Deal Days" will let you make up for that. The sale
has brought back many of the same steep discounts we saw during the summer as well as new sales. We've also put together category roundups for deals on
headphones, tablets, TVs, video games and other gadgets.
</item 82>

<item 83>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management. My palms have gotten toasty playing the OLED
model, but they haven’t broken a sweat like they often do with the original. The new fan may be larger, but it’s not any louder, and it clearly works a treat.

The Steam Deck OLED has a new Wi-Fi 6E module and improved antennas that should enable faster downloads for anyone with a compatible router. I don’t have a 6E
router or multi-gigabit Wi-Fi, but with a regular Wi-Fi 6 router, the top download speed I saw on the new Steam Deck was 562mbps, which was about 10 percent
faster than the old Steam Deck, and 10 percent slower than the speeds I get on my MacBook Pro. The device uses Bluetooth 5.3 and adds a dedicated antenna, which
should improve functionality when using Wi-Fi and Bluetooth at the same time.
</item 83>

<item 84>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
The improved 90 Hz refresh rate falls short of the Asus ROG Ally’s 120 Hz, though it’s worth considering that the higher the frame rate, the faster the battery
drain — assuming you can even spot the difference between 90 Hz and 120 Hz.

You know what’s more interesting than a list of technical specs? Actually seeing the screen. It’s gorgeous. On Sunday morning, I picked at Sega’s Like a Dragon
Gaiden: The Man Who Erased His Name while watching football and Christmas movies with my family. The new Yakuza game looks fantastic on high settings at 60 fps,
its night sequences flaunting the OLED’s dark blacks and the abundance of neon signage showcasing the HDR.

I got so distracted pulverizing bad dudes that it took me some time to notice the Steam Deck OLED’s most welcome feat. When playing AAA games on the original
Steam Deck, I had always kept the handheld plugged into an outlet. But with the OLED model, a couple of hours had passed before I realized I’d forgotten to
connect the charger.
</item 84>

<item 85>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This was a year of upheaval in video games. The industry has shapeshifted over the past 12 months, and it’s not all due to Microsoft’s lengthy acquisition of
Activision, Blizzard and King. While Xbox executives were defending the legality of a $69 billion deal that would create the third-largest video game studio in
the world, smaller companies were firing staff and shutting down entire teams, even amid fervent collective-bargaining efforts. It’s been a wild ride.

In 2023, the main factors molding the video game landscape were consolidation, layoffs and unionization, with each of these phenomena feeding into each other.
This past year, the video game industry shrank, even as it grew financially.

Consolidation

When Microsoft's purchase of Activision-Blizzard-King was finally approved on October 12, 2023 after nearly two years of regulatory hurdles, it became the
world’s third-largest video game studio by revenue.
</item 85>

<item 86>
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
Unlike the last-gen model, the 2022 iPad sports a larger 10.9-inch screen, USB-C support, and a faster processor. Read our review.

The best ebook reader deals

The 2019 Kindle Oasis with 8GB of storage and ads is available for $199.99 ($50 off) at Best Buy. The 7-inch e-reader sports physical page-turning buttons and a
waterproof 300ppi E Ink display but lacks the USB-C support found in newer Kindles. Read our review.

The best smart speaker deals

The best smart display deals

Echo Show 5 (third-gen) $ 50 $ 90 44 % off $ 50 $ 50 $ 90 44 % off The new Echo Show 5 is just as small as its predecessor but features updated mics and a
speaker system that delivers double the bass and clearer sound quality.
</item 86>

<item 87>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
Engadget

Hades is my most-played game on the original Steam Deck, so I used that to test out Valve’s battery life claims. Though it doesn’t support HDR, a game like
Hades pops on the OLED, and its animations look more fluid than on the LCD version. The OLED lasted four hours and 23 minutes before dying. With the same
settings (800p with a 60 fps cap and similar brightness) the original model died after two hours and 42 minutes. That’s a 62 percent improvement, surpassing
Valve’s own estimates, though batteries degrade and I’ve had my LCD Steam Deck since the device’s launch in early 2022. In a stress test that involved playing
Elden Ring with everything maxed out, the OLED battery lasted two hours and 20 minutes, about 40 percent longer than the original.

Playing the Steam Decks back-to-back highlighted another significant quality-of-life improvement: heat management.
</item 87>

<item 88>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

There are still tons of Apple Black Friday deals you can shop for online; these are our favorites.

Most of Apple’s current-generation devices went on sale for Black Friday — and most of those deals look like they’ll carry over through the weekend and into
Cyber Monday. That means if you were otherwise occupied on Friday, you can still save on iPads, Apple Watches, MacBooks, AirPods and more. Some of the prices
are new lows and many others match the record prices we’ve seen previously. We’ve tested and reviewed nearly every device below, so check out the links to our
full reviews and guides if you need help deciding which deals to snag. These are the best Apple Black Friday deals you can still get today from retailers like
Amazon, Target, Best Buy and B&H Photo.
</item 88>

<item 89>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
It performs best in a PC with a PCIe 4.0 motherboard or in a PS5 (with a requisite heatsink). $70 at Amazon (1TB, with heatsink)

The Western Digital WD_Black C50 Storage Expansion Card for Xbox Series X / S consoles is selling for a new low price of $124.99 ($25 off) at Amazon.

Tabletop games

Catan $ 27 $ 49 45 % off $ 27 $ 27 $ 49 45 % off The iconic board game classic that has players strategically competing and trading for resources. $27 at
Amazon$29 at Target

Happy Salmon is selling for just $8.63 ($4 off) at Amazon. The kid-friendly party card game by the Exploding Kittens creators is something you can play to
completion in just a few minutes. It’s a great ice breaker for adults and children alike, getting everyone out of their seats and quickly causing a raucous.
</item 89>

<item 90>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$400 at Amazon (refurbished)

Jsaux Mod Case for Steam Deck $ 24 $ 30 20 % off $ 24 $ 24 $ 30 20 % off A protective modular case system for Valve’s Steam Deck. It consists of a base case
with grippy texture, a built-in kickstand, a removable lid, and a rear mounting system that can attach a power bank using an adjustable strap or Jsaux’s add-on
cooling fan. $24 at Amazon

Jsaux’s 45W USB-C charger with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a
Nintendo Switch or just about any USB-C phone or tablet. It’s just $15.19 ($3 off) right now at Amazon.

with a five-foot cable and included right-angle adapter may be designed for the Steam Deck, but it’s also a great charger for a Nintendo Switch or just about
any USB-C phone or tablet.
</item 90>

<item 91>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
$59 at Amazon

Storage for consoles and PCs

A massive 1TB SanDisk Extreme microSD card is now $89.99 $79.99 ($13 off) at Amazon. That can store a whole lot of games on a Steam Deck or Nintendo Switch.

Samsung Evo Select microSD card (512GB) $ 30 $ 40 25 % off $ 30 $ 30 $ 40 25 % off This Samsung Evo Select microSD card offers 512GB of extra storage for a
Nintendo Switch, Steam Deck, or Asus ROG Ally at a good price point. $30 at Amazon

Western Digital WD_Black SN850X SSD $ 70 $ 180 61 % off $ 70 $ 70 $ 180 61 % off The WD_Black SN850X is one of the fastest M.2 drives around, with a sequential
read speed of up to 7,300 MB/s and a write speed of up to 6,600 MB/s.
</item 91>

<item 92>
The 26 best Amazon Prime Day deals you can still shop for today (Engadget, 2023-10-12T15:44:16+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Amazon's October Prime Day sale officially ended on Wednesday, but a few of the deals are still live as we head into the weekend. If you were hoping to check
off a few more names from your holiday gift list or buy something for yourself at a discount, see what remains on sale below. We combed through our coverage
over the past few days to find the best of what's left. While Amazon has returned most of their own devices, like Echos and Kindles, back to their full prices,
a handful of products from other brands like Google, Samsung and Anker, have maintained the lows they hit on Prime Day. Here are the best deals from Prime Big
Deal Days that you can still get right now.
</item 92>

<item 93>
The best anti-Prime Big Deal Day deals (The Verge, 2023-10-10T22:24:13+00:00)
This site is protected by reCAPTCHA and the Google Privacy Policy and Terms of Service apply.
</item 93>

<item 94>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
You know, we expanded the company to nine times the size we were before Fortnite took off. And so we’re just trying to forge the strategy where we’re living
within our means and doing everything we absolutely can.

But we’ve gotten so much traction with Fortnite in so many different ways, and with the Epic Games Store, right, we’re going to continue investing heavily here.
The Epic Games Store is the untold success story in the background. Since it launched in 2018, we now have 80 million monthly active users. Steam has 120
million, so we’re catching them fast! Fortnite has more concurrent users right now than all Steam games combined. You should really expect dramatic changes that
benefit all developers over the next few years.

[Editor’s note: Sweeney isn’t right about Fortnite having more concurrent users than Steam. Epic spokesperson Natalie Muñoz confirmed Sweeney misinterpreted a
Polygon story about how Fortnite hit 7.6 million players at a time when the entirety of Steam had 10.1 million.
</item 94>

<item 95>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We know that because they came to us and told us they were doing that. And we know it because after all that went down, I talked to an Activision employee who
was close to that effort, working with the team building the store, and he reported that the entire store team Activision had employed to build their competing
store had been disbanded as soon as Activision signed their deal with Google.

[Editor’s note: We did not hear from such a person at trial; I’d be eager to speak to them now!]

We know Activision was telling Google they were building their computing store; we know in Google internal discussions they said they didn’t want Activision
building a competing store. They agreed to sign this deal, and they were gleeful about the fact they dissuaded perhaps their number one competitor at the time
from launching their own store.

Four years later, Microsoft acquired Activision Blizzard — and one of the big talking points from Microsoft to the European Union was that the merger
strengthened the company overall in order to provide a viable computing store on iOS and Android.
</item 95>

<item 96>
The best October Prime Day gaming deals (The Verge, 2023-10-10T12:30:00+00:00)
Amazon’s fall Prime Day event is underway. For more deals we recommend across all categories, check out our full roundup here.

Amazon Prime Day is one of the best times of the year to save on a wide variety of video games. Titles released in the last few months are getting some decent
discounts (which still beats paying full price), and many older games from the last few years fall to their best bargain-bin prices (which is great for stocking
up on those “Man, I’ve been meaning to get around to that one” games).

Related The best October Prime Day deals available on Day 2

Here, we’ll go over all the best gaming deals we can scrounge up during Prime Big Deal Days across PlayStation 5, PS4, Xbox Series X / S, Nintendo Switch, and
PC. We’re also on the lookout for hardware deals on things like controllers, streaming peripherals, and the consoles themselves.
</item 96>

<item 97>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The first thing to know about the Steam Deck OLED is that it’s not the Steam Deck 2. The Steam Deck OLED is a mid-cycle refresh from Valve, similar to the
Switch OLED upgrade, but there’s a lot more going on internally here. Its screen is better, its battery life is better, its thumbsticks are better, the haptics
are better, and its chip and thermals are better. One thing that hasn’t changed is its form factor — this is still one of the biggest babies on the handheld
block.

The new Steam Deck’s main upgrade is its OLED screen, which replaces the original LCD and, crucially, has native support for HDR and faster frame rates. The
OLED also has smaller bezels, making the display larger at 7.4 inches, compared with the original 7-inch screen.
</item 97>

<item 98>
The Steam Deck OLED improves much more than its screen (Polygon, 2023-11-09T18:00:00+00:00)
When we award the Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth
fitting into your schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .

It appears, at first glance, that Valve cribbed from Nintendo’s playbook with the Switch OLED, for which Nintendo upgraded the device’s screen but not its guts.
In terms of raw power, the Steam Deck OLED is identical to the original Steam Deck. Valve says this synchronicity is meant in part to continue to provide
developers with a single goal when optimizing their games — though it promises that the Steam Deck 2 (which the company hopes to release in two to three years)
will have a generational power leap.

For those who haven’t tried a Steam Deck, though, I wouldn’t wait for its sequel. Because while Valve hasn’t upped the horsepower, it has improved practically
everything else.
</item 98>

<item 99>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
That's a $30 discount. Both the USB-C and Lightning models are on sale; the latter is compatible with Apple's iPhone 15 series as well as numerous Android
phones. We gave the Backbone a positive review last year, and we recommend it in our guide to the best gaming handhelds: It's more cramped than a traditional
gamepad, naturally, but it has all the requisite buttons to play modern games, it's extremely easy to set up and it works well for remote streaming from your
console or PC. If nothing else, it's a much better value than the PlayStation Portal.

PlayStation, Xbox and Switch video game sales

If you're all set on the hardware front, a truckload of PlayStation 5, Xbox Series X/S and Nintendo Switch games have dropped in price for Black Friday too.
Both Sony and Microsoft have rolled out holiday sales on their respective digital storefronts, while a number of physical games are down to new lows at Amazon,
Best Buy and others.
</item 99>

<item 100>
Steam Deck OLED review: It's just better (Engadget, 2023-11-09T18:00:38+00:00)
The updated display is smooth, crisp and bright, like a perfect lemon tart. (Even then, I do not recommend licking the screen.)

Battery life

One of the downsides of the first Steam Deck was its short battery life, which maxed out at eight hours, and that was only under ideal conditions. Valve says
the new Steam Deck battery will last 30 to 50 percent longer, depending on how you’re playing, and it charges faster — the company claims it will go from 20
percent to 80 percent in 45 minutes.

Custom AMD APU

The Steam Deck OLED has a die-shrunk version of the custom AMD APU found in the original Steam Deck — same CPU and GPU, just more efficient. This is the process
Sony takes advantage of to build the PS4 Slim and PS5 Slim, but instead of collapsing the Steam Deck around smaller hardware, Valve chose to add a bigger
battery and larger fan to its existing chassis.
</item 100>
=== END ITEM id=c0545e81d5b4dcf6 ===

=== ITEM id=9b5f88cbf6c0b25f ===
Gold sentences and facts:
Sentence 0: Love Story This finally brings us to the fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No.
  Fact 0.0: This sentence follows the heading 'Love Story'.
  Fact 0.1: This finally brings us to the fated sighting of Swift at Arrowhead Stadium on Monday.
  Fact 0.2: Swift was cheering on No.
Sentence 1: The game is taking place at Arrowhead Stadium, where Swift has performed herself and attended a game before.
  Fact 1.0: The game is taking place at Arrowhead Stadium.
  Fact 1.1: Swift has performed at Arrowhead Stadium.
  Fact 1.2: Swift has attended a game at Arrowhead Stadium before.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has shared that she has nothing to hide.
  Fact 2.1: Swift has a new relationship with Kelce.
  Fact 2.2: Swift has said she has nothing to hide in her new relationship with Kelce.

Retrieved items:

=== END ITEM id=9b5f88cbf6c0b25f ===
