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

=== ITEM id=b325c467186b2b69 ===
Gold sentences and facts:
Sentence 0: If you've got your gift list handy, now's a good time to start checking it off so you won't be pressed for time or funds when Black Friday and Cyber Monday come around next month.
  Fact 0.0: It is a good time to start checking off a gift list now.
  Fact 0.1: Black Friday and Cyber Monday come around next month.
  Fact 0.2: Checking off the gift list now means the reader won't be pressed for time or funds when Black Friday and Cyber Monday come around.
Sentence 1: “One of the hardest things is resentment can build up if you feel like you need to buy gifts for your family and you can barely make your student loan payments, so bring things up,” Alev said.
  Fact 1.0: Alev said one of the hardest things is that resentment can build up if you feel you need to buy gifts for your family and can barely make your student loan payments.
  Fact 1.1: Alev said people should bring things up.

Retrieved items:
<item 1>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
This can help the housing market retain some of its momentum without falling apart.”

Bank of America economists believe that low inventory of existing homes for sale combined with solid home sales to millennials could help put a “floor beneath
home prices,” but that doesn’t mean there won’t be some near-term pain due to higher mortgage rates.

“With rates likely staying higher for longer, we are cautious of potential turbulence ahead,” the group warned. Bank of America has previously forecast 0% home
price growth and falling home sales for the full year 2023, but it didn’t offer a new prediction in its latest note.

Eventually, as inflation fades leading the Fed to cut interest rates, housing affordability will improve. “At that stage, we should see a more stable and
healthy housing market,” Park and her team wrote. “Until then, hang tight, it may be a bumpy ride.”

This article has been updated to correct a mistaken term used to refer to Jeseo Park.
</item 1>

<item 2>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
In a recent poll from the American Psychological Association, 41 per cent of US adults said their stress levels increased from November to January when compared
to other months of the year. Nearly 50 per cent of Americans also described their stress level during the season as “moderate”.

Everyone has different concerns leading up to Christmas, Hanukkah, or New Years - from family drama at the dinner table to a small spats with your mother about
decorations. With what can be stressful events, it only begs the question: do you really want to bring your partner into the mix? While it’s important to
eventually introduce your loved ones to your significant other, relationship experts believe there are some discussions you should have beforehand, especially
during the holidays.

Speaking to The Independent, Rachel Zar - a licensed marriage and family therapist at Spark Chicago Therapy - acknowledged that while every family is different,
there’s a variety of problems that could arise during the holiday season - mainly because we put too much pressure on having a joyful holiday.
</item 2>

<item 3>
The best apps and tools for managing your money online (The Verge, 2023-11-12T13:00:00+00:00)
got some love, too. The manual way can be the way… but it’s more work. Some of you love having custom-built systems that you can tend to for a few minutes a
week, and there’s certainly no beating “it’s just a file on your computer that you own.” But almost everyone who recommended this strategy also said it’s a lot
of work and can be too easy to give up on.

Some of you love having custom-built systems that you can tend to for a few minutes a week, and there’s certainly no beating “it’s just a file on your computer
that you own.” But almost everyone who recommended this strategy also said it’s a lot of work and can be too easy to give up on. Or maybe just use your bank. A
lot of banks now offer budget tools from right within your banking app, so you can manage your money right next to where it lives. (I hear good things about
SoFi’s features on this front.)
</item 3>

<item 4>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
Between being a mom and a new grad with a salary to match, Plauché says she can barely manage her daily expenses. Still, she feels a responsibility to give her
family gifts because that’s how they have traditionally shown each other love.

“I don’t want to give my family this message that they’re not loved and it definitely adds a lot of pressure,” she said.

Johnson feels guilty because the holidays are the time she feels she can show her family how grateful she is for them. She has always prided herself on being a
great gift giver who not only finds something nice but knows how to spot unique and special gifts.

“I think I’m a little embarrassed to tell them that this year I might not be able to be a good gift giver,” said Johnson.

Acknowledging that you’re feeling pressure is a good first step towards making a plan to set expectations and be realistic about what you can spend.
</item 4>

<item 5>
Fast fashion retailer ASOS struggles to engage consumers as company losses balloon (The Independent - Life and Style, 2023-11-01T15:36:20+00:00)
After all, the company is known for its next day delivery and easy returns service.

Though many of us are reliant on the online fast fashion retailer for last-minute holiday purchases or big seasonal shops, a new report shows that ASOS’s
pre-tax losses ballooned to almost £300m in the year to 3 September, compared with £31m the previous year, according to the results published on Wednesday (1
November).

Despite its huge popularity, the once undisputed queen of online retailers now faces stiff competition from abroad, challenging economic headwinds, and a
maturing clientele potentially seeking a more tailored experience. Leading some to ask, is ASOS in SOS?

When ASOS was launched in 2000, the online retailer was considered revolutionary. It was like a supersize department store at your fingertips and became the
first online retailer that UK customers could visit and find hundreds of dresses for £30 all in one place.

But, in recent years, the company has been cutting its stock as it faces severe profit and revenue losses.
</item 5>

<item 6>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
Free next-day delivery is available on select items through December 23rd.

You can get free same-day delivery if you order your package on December 23rd or December 24th. You must place your order by 12PM local time for free same-day
delivery by 7PM.

You can also order online by 5PM local time on Christmas Eve for an in-store pickup at 7PM.

Best Buy’s shipment arrival dates may vary depending on the kind of item that you’re getting. Major appliances may take longer to be delivered at this point.

Illustration by Kristen Radtke

Walmart offers two-day shipping on items sold directly from Walmart, with orders over $35 shipping for free. You must place orders by 2PM local time on December
21st to receive the package in time for Christmas, however. Next-day shipping is also available in some areas.

Walmart also offers free next-day and two-day shipping to Walmart Plus subscribers with no order minimum.
</item 6>

<item 7>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
FedEx 2023 Holiday Shipping Deadlines Final day to ship Ships via this service December 19th FedEx Express Saver December 20th FedEx 2Day and 2Day A.M. December
22nd FedEx SameDay

FedEx’s same-day shipping can get your package where it needs to be by Friday, December 22nd. That’s the last day that FedEx can guarantee that your package
will arrive by Christmas. If you ship it by Wednesday, December 20th, you can opt for the two-day shipping method. If you manage to get it to FedEx by Tuesday,
December 19th, you can go the Express Saver route and (hopefully) pay a reasonable fee for shipping.
</item 7>

<item 8>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
Follow us!

Gifts for the bookworms

Obviously, you could just gift a bibliophile a book and they’d probably be happy. But what if you don’t know what your giftee is into or simply want to give
them more options? In that case, a gift card to their favorite bookstore or a subscription to something like Kindle Unlimited, which grants members access to
millions of ebooks and even select audiobooks, is a good idea. That said, we’ve rounded up some of our favorite alternatives below.

For fans of Marvel and DC comics, as well as manga like Fairy Tale, a Comixology Unlimited subscription is perfect. For $5.99 a month, they’ll be able to enjoy
over 45,000 comics and graphic novels, not to mention more than 2,400 manga titles from their phone or tablet. Subscribers also get discounts on select digital
books.

is perfect.
</item 8>

<item 9>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
In the consumer electronics space, we expect most companies to have some sort of Black Friday promotion, so that means you will likely be able to get big-ticket
items from brands like Apple, Samsung, Google, Sony, LG, Sonos and others for much cheaper prices.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.
</item 9>

<item 10>
The 30 best Amazon Prime Day deals for October 2023 (Engadget, 2023-10-10T17:41:37+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Here are the best Prime Day deals we could find gathered in one place.

Amazon's second Prime Day-like sale event for 2023 kicked off in the wee hours this morning, and it's turning out to be a great opportunity to get holiday
shopping done early. Or, if you missed out on any of the deals from Prime Day in July, this year's "Prime Big Deal Days" will let you make up for that. The sale
has brought back many of the same steep discounts we saw during the summer as well as new sales. We've also put together category roundups for deals on
headphones, tablets, TVs, video games and other gadgets.
</item 10>

<item 11>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
If you are looking for a budget template, Microsoft Office offers a holiday spending-focused version.

COMMUNICATE WITH YOUR LOVED ONES

If you’re experiencing a lot of financial pressure this holiday season, Alev recommends that you try to communicate your feelings to your loved ones.

“One of the hardest things is resentment can build up if you feel like you need to buy gifts for your family and you can barely make your student loan payments,
so bring things up,” Alev said.

Sharing your financial struggles can be hard. If you want to talk but don’t want to share every detail of your financial struggles, that’s okay. Ultimately,
what you want is for your loved ones to understand how you’re feeling.

Plauché just recently started sharing her money worries with her family.

“I have been open with my family, just that I’m not going to be able to give gifts that year,” she said.

EXPERIENCES VS.
</item 11>

<item 12>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
workforce, but they’re already radically redefining the meaning of work.

Gen Zers want a sense of purpose, so they prioritize environmental, social, and governance (ESG) in the workplace, which encompasses their companies taking
accountability for their sustainability and environmental impacts, providing education and awareness for social issues, and ensuring diverse and inclusive
boards and teams.

And it tracks: Roughly two-thirds of Gen Zers say they frequently speak about important societal issues while at work, according to the Edelman report. They’re
also influencing their older coworkers when it comes to areas like work-life balance, fair pay, and employers’ involvement on social issues.

Gen Z’s financial stress

But don’t forget, Gen Zers have also lived through a global pandemic that shuttered the world for nearly two years, two recessions, and a mounting student debt
crisis, leaving them with little savings but an abundance of financial despair.
</item 12>

<item 13>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
If you can’t pay your balance in full every month, it’s important that you understand when you payment is due and have a plan to avoid paying a lot of interest.

If you’re planning to use “ buy now, pay later services ”, it’s best to be very intentional, Alev said.

“It’s easy to get caught up. It’s easy to use it a lot and have several different payments that will come down the road,” she said.

AFFORDABLE ALTERNATIVES

There is more than one way to find the “perfect” gift for your family and it doesn’t always involve spending a lot. In her quest to find gifts that don’t cause
her financial stress, Johnson has been getting creative.

“I’ve definitely been trying to save or just make gifts because I don’t really have the disposable income to buy other people gifts or even buy myself a
Christmas gift,” Johnson said.
</item 13>

<item 14>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
Retailer memberships will pay for themselves (eventually)

To get early access to Black Friday and Cyber Monday deals — or to get access at all — some retailers offer memberships. Most are familiar with Amazon Prime,
which offers a free 30-day trial and unlocks special prices on top of free shipping and other perks, but there are others. Costco is another popular example
that’s been around for a while. Its $60 membership pays for itself quickly with holiday purchases, especially if you buy food products in bulk (this is outside
of Polygon’s purview, but you cannot beat its deal on Kerrygold butter).

Walmart Plus costs $99 per year ($50 for a limited time), and serves as a ticket to get in on deals before other Walmart shoppers, plus other perks. Note: You
won’t get early access to deals with its free 30-day trial; you must be a paid member to earn access, so hit “Skip trial for Early Access” when signing up.
</item 14>

<item 15>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Adobe Analytics’ figures are based, it says, on 1 trillion visits to U.S. retail sites, 100 million SKUs and 18 product categories. Salesforce says it taps data
from 1.5 billion consumers in its research. (Both companies work with a number of giant and smaller retailers, so they have the infrastructure to provide this
kind of intel to these customers on an ongoing basis.)

Breaking out some of the trends in how people are shopping:

— Overall, desktop sales are still exceeding other kinds of screens when it comes to conversions and number of items purchased, but mobile is the standout
device in the evening hours, when it accounted for a whopping 59% of all online sales (likely because people were using that time to socialize and buy on the
sly).
</item 15>

<item 16>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
Peloton subscription (monthly subscription) $ 13 Peloton’s $12.99 monthly subscription offers thousands of streamable fitness classes revolving around a variety
of different workouts, ranging from strength training to stretching. It’s a service that’s available to everybody, so you don’t need to own a Peloton device to
use it. $13 at Peloton

A gift card for Headspace , a popular mindfulness app, offers members access to hundreds of expert-taught meditations, each designed to help them relax, sleep
better, and improve their mental health. A subscription will run you $12.99 monthly or $69.99 annually.

, a popular mindfulness app, offers members access to hundreds of expert-taught meditations, each designed to help them relax, sleep better, and improve their
mental health. A subscription will run you $12.99 monthly or $69.99 annually.
</item 16>

<item 17>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
Indeed, existing-home sales prices topped $306,000, a 5% increase since the year began, according to the Case-Shiller U.S. National Home Price Index.

“As has been the case throughout this year, limited inventory and low housing affordability continue to hamper home sales,” NAR chief economist Lawrence Yun
said in a statement. “The Federal Reserve simply cannot keep raising interest rates in light of softening inflation and weakening job gains.” The Fed has raised
rates several times this year, with Chair Jerome Powell arming his “higher for longer” stance.

Declining housing inventory levels also contribute to the drop in housing transactions. Housing inventory is down 8.1% from this time last year, according to
NAR. Some real-estate experts and economists argue that housing affordability is even worse now than during 2008, when a damaging downturn in the U.S.
</item 17>

<item 18>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
Similar to Walmart and Amazon, be aware that items at Target shipped or sold by third-party retailers can result in longer lead times. Product listings usually
spell out your options for pickup, same-day delivery, and standard shipping — complete with estimated availability times — before you add them to your cart.
</item 18>

<item 19>
Nike misses revenue expectations for the first time in two years, beats on earnings and gross margin (Cnbc | World Business News Leader,
2023-09-28T20:31:00+00:00)
Jefferies conducted a survey on U.S. consumer spending and found 54% of respondents plan to spend less on apparel and accessories. Meanwhile, 46% plan to spend
less on footwear, which doesn't bode well for Nike.

It's still too early to gauge the impact of student loan payments on Nike. Its first quarter ended in late August, and payments aren't set to resume until
October.

During the quarter, footwear sales rose 4% to $8.4 billion, making up about 68% of Nike's total sales. Apparel was down 1% at $3.4 billion.

Correction: Nike's gross margin fell 0.1 percentage points. An earlier version of this story misstated that figure.
</item 19>

<item 20>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
Then you have Best Buy’s My Best Buy Plus and Total memberships, costing $49.99 and $179.99, respectively. As for what each offers, check out Best Buy’s page
that spells it all out. This page details all of the exclusive deals, which can help you quickly recoup the cost of the membership.

Amazon Prime members get free games and DLC all year

Amazon offers freebies for Prime members ’round the calendar year, not just during the holiday months. Those offerings expire over time and new ones are added,
and Amazon’s suite of giveaways has improved over time.

We collect all of the latest free games and DLC that Amazon is giving away right here. To give you a taste, Rage 2, Star Wars: Knights of the Old Republic, and
Doom 3 are some of what’s up for grabs right now.

You’ll get price matching if you shop at the right time

Some major retailers are ensuring price protection on purchases right now.
</item 20>

<item 21>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
Yes, Christmas is tomorrow — and yes, we know you have yet to buy a gift. We understand that life gets busy, though, and sometimes it feels like the holidays
creep up on you out of nowhere. But before you spiral into a full-blown panic attack, take a deep breath. Luckily for you, the internet is filled with a
treasure trove of gift cards, subscriptions, and other great digital gifts you can buy as late as Christmas Day itself.

To help make your life a little easier, we’ve curated a list of some of the best digital gifts we’ve either used ourselves or gifted to our friends and family.
We’ve organized the list by interests, too, so you can find the perfect present whether your loved one is into the arts, exercise, or something else entirely.
That way, you’ll at least be able to gift something more thoughtful than a generic Amazon, Best Buy, or Walmart gift card — even if those are still totally
viable options in our book.
</item 21>

<item 22>
Fast fashion retailer ASOS struggles to engage consumers as company losses balloon (The Independent - Life and Style, 2023-11-01T15:36:20+00:00)
In July and August, the retailer said it was hit by wet during the summer months, forcing the retailer to cut its stock by 30 per cent.

Some experts now say that the holes within ASOS’s business model are becoming apparent. Rick Smith, Managing Director of business recovery firm Forbes Burton
tells The Independent that ASOS’s business model of allowing large numbers of free returns makes the company susceptible to financial losses during challenging
economic conditions.

“The buying culture within their target demographic sees several items returned for each order placed, which can make for slim profit margins,” Smith explains.
“ASOS has enjoyed a stratospheric rise since their inception, but have perhaps reached their ceiling now and need to cut their cloth accordingly, especially
given the amount of economic headwinds [such as as the cost of living crisis] in play at the moment.”

The buying habits of consumers have also shifted in the 23 years since ASOS launched. Smith says that sifting through hundreds of products has become a
painstaking and often unenjoyable task for its customers.
</item 22>

<item 23>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
In LA, for example, the threshold is over $25.

Prime members can get One-Day Delivery for eligible products through December 23rd with no minimum purchase amount or Two-Day Delivery through December 22nd.

If you’re a Prime member in select US cities, you can take advantage of two-hour grocery delivery through December 24th.

Shipping times and prices may vary depending on the product type, size, and whether it’s shipping directly from Amazon or a third-party seller, so make sure to
check the expected arrival date at checkout, then choose accordingly.

Illustration by Kristen Radtke

Best Buy 2023 deadlines

If you’re a My Best Buy Plus and My Best Buy Total subscriber, you can get free two-day shipping, and there’s no order minimum.

Other customers need to clear the $35 order threshold (before taxes but after coupons), and certain items are available for next-day and standard shipping.

December 21st at 11:30AM ET is the cutoff date for free shipping if you want your package to arrive by December 23rd.
</item 23>

<item 24>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
These days, it feels like there’s a Starbucks pretty much everywhere you look now, meaning a Starbucks gift card (Amazon, Best Buy, or Target) can help your
giftee get their caffeine fix whenever, wherever.

gift card (Amazon, Best Buy, or Target) can help your giftee get their caffeine fix whenever, wherever. Sometimes, it’s better to leave the cooking to somebody
else. For foodies, a gift card to a food delivery service like DoorDash (Amazon, Target) means they can nab some crab rangoons from their favorite Chinese
restaurant without leaving home.

Gifts for music lovers

Whether your giftee is a musician or just loves to unwind with some music, there are a lot of digital presents you can buy. We all know about Spotify gift cards
(Amazon, Best Buy, or Target), but there are also other streaming services that you can gift as a subscription, some of which we’ve highlighted below.
</item 24>

<item 25>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is almost here, but we wouldn't blame you if you've felt like it's been here for a while. What was once a one-day event filled with limited-time
deals has grown into a month-long event with a plethora of sales to sift through each day. This year, Amazon, Target, Best Buy and Walmart and several other
retailers kicked off their Black Friday deals early, with some starting as soon as late October. If you're overwhelmed by the amount of discounts showing up in
your feeds, you're not alone.



To help, we at Engadget have collected all of the best Black Friday deals you can shop right now, primarily on tech we've tested and reviewed. These are the
best tech deals you can get this year, on the gadgets we know to be worth your money.
</item 25>

<item 26>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Today’s 5.5% rate is definitely an improvement on last year’s 2.9%. But it’s nothing compared to the years preceding COVID-19, such as 2017, where we were
seeing growth of 18% or more.

Inflation is making an impact but not as much as the worry around consumer spending, said Adobe, which said that spending is being led actually by more
discounts to encourage buying, rather than less buying of more expensive products. That leads one to wonder what kind of impact that’s having on retailers’
margins.

“Cyber Week is off to a strong start with Thanksgiving driving a record $5.6 billion in online spend as consumers took advantage of strong discounts and
continued their shopping plans, virtually,” said Vivek Pandya, lead analyst, Adobe Digital Insights, in a statement. “Mobile shopping hit an all-time high, as
shoppers took to their smartphones to get the best deals during holiday gatherings, further solidifying mobile’s growing importance in e-commerce.”
</item 26>

<item 27>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
housing market set off the dominoes that became the Great Financial Crisis that defined a whole decade economically—and in some ways, our current predicament.

In 2010, the last time existing home sales were so low, the global economy was on life support and the Fed slashed interest rates down to zero to revive the
flatlining patient—fueling an “everything bubble” that inflated all kinds of assets along the way, not the least, housing. The zero-rate regime persisted until
the highest inflation in 40 years surprised many economists and prompted the massive series of rate hikes that have brought us to the once-unthinkable 8%
mortgage.

Some onlookers saw what was coming, for instance Zillow, which warned in May that housing would enter a “deep freeze” if the debt limit standoff failed to
resolve and America defaulted, as that would send mortgage rates up to 8%. Of course, the default was averted, but here we have arrived at 8% mortgage rates
anyway.
</item 27>

<item 28>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
For Nicole Plauché and her family, Christmas is the time of the year where they can use their preferred love language: gift giving. This year, however, Plauché
is worried about how much she can afford to spend on gifts. Her main struggle? Student loan payments.

“It just doesn’t leave much room for anything outside of just basic necessities,” said Plauché, a 23-year old tech sales manager from Dallas, Texas.

Marisa Johnson, 26, has approximately $145,000 in student loan debt from both her bachelor’s and master’s degrees. Since the COVID-era payment pause ended in
October, she’s been paying back $300 a month while she waits to see if she’ll be enrolled in a new income-driven repayment plan.

Because all of her money is going to necessities and paying back her student loans, Johnson hasn’t been able to spend as much on gifts as in previous years.
</item 28>

<item 29>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
29. “These headwinds are causing both buyers and sellers to hold out for better circumstances.”

But with the development that home sales transactions are at their lowest level in the past 13 years coupled with century-high mortgage rates of 8%, many real
estate experts and economists aren’t hopeful that affordability constraints will let up soon.

Out of fear of losing their lower interest rates, current homeowners are resistant to putting their homes on the market, ultimately leading to the abysmal
inventory levels—leaving fewer and fewer properties exchanging hands. Indeed, more than 90% of existing-home owners are locked into mortgage rates below 6%,
Odeta Kushi, deputy chief economist at First American, previously told Fortune.

“These homeowners do not have a financial incentive to sell,” Kushi says. “The combination of reduced affordability and an even stronger rate lock-in effect
suppresses home sales because you can’t buy what’s not for sale, even if you can afford it.”
</item 29>

<item 30>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
The best deals of the year were once known to occur during Black Friday and Cyber Monday, which in 2023 take place on Friday, Nov. 24, and Monday, Nov. 27,
respectively. Not anymore!

In 2023, the best Black Friday deals kicked off on Nov. 17 at Amazon, Best Buy, Walmart, and other retailers. Instead of hosting all of the doorbuster deals on
one day right after Thanksgiving, retailers see the value in maximizing the amount of time you have to shop for products before Black Friday. And I’ve gotta
say, it’s quite nice to not feel rushed while shopping.

In addition to presenting you with the best board game deals and price drops on gaming monitors, plus deals on SSDs and microSD cards, there are some bits of
info and wisdom I’ve organized into sections below that might help you feel more informed as you begin to shop for the holidays.
</item 30>

<item 31>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
If you know somebody trying to learn how to code, a subscription to the coding educational platform Codecademy ($29.99 per month) can help them build their
portfolio with online courses, a community, fun events, cheat sheets, and other resources. There’s even a plan for those looking to change career paths, one
that offers all the above as well as technical interview help. That latter is currently available for $120 a year (half off) when you use promo code BYE2023 .

($29.99 per month) can help them build their portfolio with online courses, a community, fun events, cheat sheets, and other resources. There’s even a plan for
those looking to change career paths, one that offers all the above as well as technical interview help. That latter is currently available for $120 a year
(half off) when you use promo code .
</item 31>

<item 32>
Earnings week ahead: DocuSign, NIO, Broadcom, GameStop and more (Seeking Alpha, 2023-12-03T13:02:00+00:00)
The first full week of December features several significant companies slated to announce their quarterly results. These earnings releases hold considerable
implications for various sectors, including energy, software, retail, autos and semiconductors.

Among the companies scheduled to release financial figures are NIO (NYSE:NIO), GameStop (NYSE:GME), DocuSign (NASDAQ:DOCU), ChargePoint (NYSE:CHPT), Dollar
General (NYSE:DG), C3.ai (NYSE:AI) and Broadcom (NASDAQ:AVGO). Additionally, reports are expected from Toll Brothers (NYSE:TOL), Campbell Soup (NYSE:CPB),
AutoZone (NYSE:AZO), Lululemon (NASDAQ:LULU), MongoDB (NASDAQ:MDB), Signet (NYSE:SIG) and Chewy (NYSE:CHWY).

Below is a curated list of earnings due in the weeks of December 4 to December 8:

Monday, December 4

GitLab Inc.
</item 32>

<item 33>
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
</item 33>

<item 34>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
With this in mind, parents should avoid "processing their frustration around the holidays audibly around the children," she said.

"The ‘mom brain’ is wired to process problems rapidly and move on to solutions," she said. "The ‘child brain,’ however, is wired to process a problem internally
and take on the responsibility."

Said Keeley, "Knowing this, make sure you don’t do any ‘grinchy griping’ out loud to the children. You may see it as temporary venting, but they see it as, ‘Mom
is upset and I’m probably the reason.’"

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Instead, parents should stay calm, find a reason to be grateful and vocalize it, even when frustrations accumulate.

5. Neglecting the ‘give list’

While building a wish list is fun for kids, Keeley emphasized the moral importance of the "give list."
</item 34>

<item 35>
Should you introduce your partner to your family during the holidays? Relationship experts weigh in (The Independent - Life and Style,
2023-12-12T16:02:44+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

If you’re in the midst of a new relationship this holiday season, you may be faced with the opportunity to introduce your significant other to your family.
However, it’s likely you’ve found yourself mulling over an array of reasons why the holidays may not be the right time for such an introduction.

The holiday season is often perceived as the most wonderful time of year, but that’s not necessarily the case for everyone.
</item 35>

<item 36>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
She’s been looking into gifting a used book to her dad and because she loves painting and already has all the tools, she’s working on a painting for her mom.

Other affordable alternatives can be sending hand written notes to your loved ones, buying from thrift stores, hosting a cookie bake off or simply spending
quality time with your family.

___

The Associated Press receives support from Charles Schwab Foundation for educational and explanatory reporting to improve financial literacy. The independent
foundation is separate from Charles Schwab and Co. Inc. The AP is solely responsible for its journalism.
</item 36>

<item 37>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
“Consumers are embracing mobile wallets to break down friction between discovering on social and purchasing on mobile.”

— Retailers are really pushing out discounts to get people more willing to spend money this year. Both Adobe and Salesforce said in the U.S. discounts were up
by about 28%. Categories that saw dramatic markdowns included toys, electronics and computers, per Adobe’s figures; respectively these saw sales up by 182% and
113% over last month.

— Buy Now Pay Later (BNPL) remains a popular option for paying upfront. BNPL drove $390 million in online spend, Adobe said, up 7.5% on last year.

We’ll update with more data later, and with Black Friday figures as they start to emerge.
</item 37>

<item 38>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
If you’re not, consider signing up. Here are a few highlights from this week:

Bitcoin surge: Jacquelyn writes about Bitcoin’s rapid-fire ascent to $44,000, which came on the back of roughly 25% gains in the last week. Her piece for TC+
explores what’s driving Bitcoin’s price ascent and similar value gains among other tokens — and whether the good vibes continue into the new year.

To swap, or not to swap: Tim reports on how consumer EV battery swapping could usher in freedom for a wide range of people, allowing them to participate in the
EV transition in ways that traditional built-in batteries don’t. The challenge is making the unit economics work.

Coinbase and Robin and the future of fintech: Investors are betting that consumer trading of equity and crypto is rebounding and are consequently pushing the
value of some former startups higher, Alex writes. That could spell good news for startups offering consumer trading services directly — or indirectly, for that
matter.
</item 38>

<item 39>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
For giftees into arts and crafts, a Craftsy membership ($113 a year) grants access to more than 2,000 live and on-demand classes led by experts covering
everything from baking and cake decorating to woodworking and painting. Members also get to connect with other crafters in the Craftsy community and attend live
events.

Gift cards for pretty much anyone

Sometimes the best gift card is one that’ll give your giftee a ton of options, especially if you’re having a hard time figuring out what they want. Gift cards
from major retailers like Amazon, Best Buy, Walmart, and Target are perfect in these situations, namely because they’ll let your giftee choose whatever they
like from a wide range of departments.
</item 39>

<item 40>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
Black Friday and Cyber Monday may be in the past, but some sales are still hanging around (although maybe not for long). For more leftover deals we recommend
across all categories, be sure to check out the rest of the Cyber Monday still available here.

Every month or so, we like to ask our staff about their favorite stuff — whether it’s pet toys, travel aids, kitchen gadgets, or straightforward tech. And the
results are usually very different, very interesting, and a lot of fun.

In celebration of the annual post-Thanksgiving sales, we looked through some of our recent “favorites” articles and found deals on a lot of the tech, kitchen
tools, travel aids, and pet toys we like. We thought we’d list a few in case you’ve read about them in the past and thought, “Well, that sort of sounds good,
but it’s a bit pricey.”
</item 40>

<item 41>
8 authors and their 8 gift book picks for this holiday season (TechCrunch, 2023-12-01T16:05:15+00:00)
8 authors and their 8 gift book picks for this holiday season TechCrunch talked to eight authors with ties to the startup community to see which books they
would gift this year

Book blurbs have become a central part of the publishing industry: Who better to endorse a book than other authors and thought leaders? In that same spirit, we
asked several writers to recommend books that you and other TechCrunch readers may want to gift this holiday season.

Read on for recommendations from:

The responses have been lightly edited for length and clarity. This article contains links to affiliate partners where available. When you buy through these
links, TechCrunch may earn an affiliate commission.

Jon Evans

You may know Jon Evans as a former TechCrunch columnist or from his engineering and CTO roles, but he is also a novelist. Most of his books are technothrillers
painting travelers caught in tricky situations in exotic locations.
</item 41>

<item 42>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
“I’ve mostly been trying to think of affordable gifts or combining gifts with my older sister,” said Johnson, a non-profit worker in Citrus Heights, California.

Plauché and Johnson are just two of the millions of people who had to start paying back student loans after the Biden administration ended the three-year
payment pause.

“We know that the holidays are often a very joyful time of the year and an expensive one every year,” said Courtney Alev, consumer financial advocate at Credit
Karma. “With federal student loan payments just resuming, it’s going to add another often very significant line item into holiday budgets.”

If student loan payments are cutting into what used to be your holiday budget, here are some recommendations:

THE PRESSURE IS REAL

From family expectations to social media and TV ads showing lavish gifts, many people feel that the only way they can show love is to spend a lot of money.
</item 42>

<item 43>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
— More on mobile: Salesforce said that for the day, mobile accounted for 79% of all online traffic globally and 82% in the U.S., and that mobile wallets were
really making a mark for reducing some of the buying friction on the smaller devices. Apple Pay saw transactions up by 47% compared to 44% up for all mobile
wallets overall. Social on mobile — think Instagram links, TikTok and Snapchat — has become an influencer in itself. These collectively accounted for 13% of all
traffic referrals to sites. Notably, they are not where the purchases are being made for the most part, though.

“Mobile traffic and sales are soaring as people are on the go once again this holiday weekend,” said Salesforce VP and GM, Retail, Rob Garf, in a statement.
“Consumers are embracing mobile wallets to break down friction between discovering on social and purchasing on mobile.”
</item 43>

<item 44>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
While you don’t need an account to see pricing information available through the app, making a profile will allow you to set up watchlists and get email price
alerts for your favorite products. You’re welcome to manually populate your own watchlist, but the app also allows you to import any existing Amazon wishlist
into your CamelCamelCamel profile. Make sure to grab its Camelizer browser extension, too.

Honey and its accompanying extension allow you to track pricing history changes on specific products, but across a number of retailers, not just Amazon. Also,
as an added bonus, Honey will also automatically search for promo codes when you check out at a supported retailer.
</item 44>

<item 45>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
Fitbit also offers guided programs covering topics like nutrition, along with the option to one-on-one with a professional health coach for an extra $54.99 per
month. For those who prefer in-person classes to virtual, a gift card to a subscription service like ClassPass will let your giftee try out thousands of gyms
and fitness studios in their local area, not to mention nearby salons and spas.

will let your giftee try out thousands of gyms and fitness studios in their local area, not to mention nearby salons and spas. If your giftee is too busy to
prepare healthy meals every day, a gift card to Fresh N Lean, Blue Apron, HelloFresh, or any meal prep service that offers a healthy selection of meal kits will
be very welcome.

Gifts for foodies

Whether they’re a diehard foodie, a wine connoisseur, or a caffeine addict, the internet is filled with subscriptions and gift cards for all types.
</item 45>

<item 46>
Nike misses revenue expectations for the first time in two years, beats on earnings and gross margin (Cnbc | World Business News Leader,
2023-09-28T20:31:00+00:00)
With no single partner representing more than a mid-single digit of Nike's total business," said Friend.

"While the ultimate landing spot of digital and direct isn't as clear, we do believe we're going to be a more direct and a more digital company, and a more
profitable company," he said. "And there's a channel mix and channel profitability opportunity that comes with that as well."

Meanwhile, inventories fell 10% to $8.7 billion. The drop was driven by a decrease in units but offset by product mix and higher manufacturing and production
costs.

"On the whole, we're very comfortable with the level of inventory in the marketplace in relation to the retail sales that we're seeing as we begin increasing
levels of wholesale sell in our second half," said Friend.

Amid decades-high inflation rates, consumers have been pulling back on apparel and footwear. With the resumption of student loan payments looming ahead, some
analysts expect those sectors to take an even greater hit.
</item 46>

<item 47>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
Individual plans cost $49.99 (Amazon, Walmart), while a family plan — which allows for up to eight accounts — is currently available via gift card at Amazon and
Walmart for $71.89 ($8 off).

. In addition to offering all the same benefits as the Switch Online membership, it also grants access to Game Boy Advance, Sega Genesis, and Nintendo 64 games,
as well as DLC content for some titles. Individual plans cost $49.99 (Amazon, Walmart), while a family plan — which allows for up to eight accounts — is
currently available via gift card at Amazon and Walmart for $71.89 ($8 off). Alternatively, you could buy a gift card to a store like GameStop (Amazon,
GameStop), which is useful if you don’t know which console your giftee prefers or want to give them the option of buying accessories and games.

Gifts for the adventurers and globe-trotters

Is your giftee in dire need of a screen break?
</item 47>

<item 48>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
But feeding friends and family is perhaps one of the most humane acts we can pursue. It’s less about the spirit of giving, and more about admitting that people
have innate, basic needs, like food and social comfort, and that those needs are best fulfilled when people work in tandem. Honestly, we should try to do it
more on every other day of the year: The number of people volunteering to help feed the homeless and families in need peaks around November and December, but
that energy needs to be carried through the preceding 10 months, too.

Spider-Man represents these needs, even though very few of us have to balance photojournalist work, a fraught dating life, and pummeling Dr. Octopus. Peter
Parker is often the most financially strapped among his Avengers buddies, and frequently the loneliest, too. Those are recognizable traits among many young
people, even those without radioactive-spider powers. When you’re growing up and trying to figure out the world, it’s easy to feel lost and isolated.
</item 48>

<item 49>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
"Many parents make the mistake of teaching their children that Christmas is a time to make a ‘wish list’ and get what they want," she said.

"Many parents make the mistake of teaching their children that Christmas is a time to make a ‘wish list’ and get what they want."

"This is loads of fun, but can often create a big case of 'the gimmes’ in your children."

She added, "Life is balance. It’s about receiving and giving."

As parents guide their kids to create a wish list, Keeley encouraged them to spend equal time creating a list of gifts to give to family and friends.

"Collaborate with them on what they want to give to the people they love, and encourage them to use their imagination and creative ability, not just your credit
card," she said.
</item 49>

<item 50>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
For those into buying books from local bookstores, there’s also a gift card for Bookshop.org .

(Barnes & Noble, Amazon, or Best Buy) might be a good idea. For those into buying books from local bookstores, there’s also a gift card for . For those who
prefer traditional newspapers and magazines, you can gift a subscription to The New York Times or The Washington Post, or publications catered to specific
interests like Cosmopolitan and National Geographic.

Gifts for the creatives

Movie buffs and bibliophiles are easy to shop for, but what do you get the creative type? It’s actually not that hard — just buy them something to help them
create, whether that’s an online course or access to a new tool. Below are a few subscriptions and gift cards creators will love that you can also buy last
minute.

A MasterClass membership (which starts at $10 a month) provides access to classes taught by world leaders and other subject matter experts, including
screenwriters, musicians, and business experts.
</item 50>

<item 51>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
The same thing happens year after year. Things get busy with work, school, and life in general until you’re suddenly up against the clock as you race your gifts
to your local post office, FedEx store, or UPS location. We’ve all been there, but we’d like to be of some help as you plan your next move and begin sending out
your holiday packages.

If you plan on sending your gifts via traditional mail, what follows are the upcoming domestic shipping deadlines for various retailers and shipment carriers so
you can ensure that your packages arrive before Christmas, Kwanzaa, or whatever holiday it is you celebrate this time of year. Hopefully, it’ll help you avoid
some holiday stress, if at all possible.
</item 51>

<item 52>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
Amazon does not price match, which is annoying, but it’s tolerable since it often has the lowest prices of all retailers anyway. Amazon only protects the price
on pre-orders that have its “Pre-order Price Guarantee,” which ensures that you’ll pay the lowest amount a product sells for before it’s launched.

Not only does Best Buy match prices from “qualified competitors,” but its site says that most products bought between now and Dec. 30 can be price-matched
through Jan. 13, 2024.

Walmart’s policy says it won’t price match with competitors, nor does it price match for products purchased online that later decrease in price.

Target says that purchases made between Oct. 22 and Dec. 24 can be price-adjusted if the cost lowers in the span of time, and if you show proof of purchase. You
may be eligible for a price match on a purchase, but the rules for submitting documentation vary depending on whether you’re shopping online or in store. It’s
all explained here.
</item 52>

<item 53>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Black Friday is often a boon for deals on headphones and earbuds, and this year is no different. The shopping holiday has brought a plethora of deals on all
kinds of headphones, include several top picks from our guides to the best wireless headphones, wireless earbuds, budget wireless earbuds and gaming headsets.
Other popular models from the likes of Sony, Beats, Bose and Apple are discounted as well. All of this means it's an ideal time to pick up a new set, whether
it's to give as a gift to someone on your holiday shopping list or just to replace your own aging cans. To help you get a good bargain, we've rounded up the
best Black Friday deals on headphones and earbuds we could find for 2023.
</item 53>

<item 54>
The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' (Fortune, 2023-11-12T17:51:37+00:00)
Consumers gearing up to buy the latest imported appliances, clothes or electronic gadgets this holiday season might want to spare a thought for the companies
that will struggle to make money for the next few years hauling products across the ocean.

That’s because the container shipping industry, cast as the Grinch that spoiled Christmas over the past two years with record-high freight rates and slow
deliveries, is returning to its pre-pandemic place in the corporate world: perennial underachiever Charlie Brown.

The biggest carriers posted net income totaling $364 billion in 2021 and 2022, according to figures compiled by industry veteran John McCown, after a decade of
scant profits. They’ll likely drift back into the red this quarter as the rates they charge fall below costs and look to stay there for the foreseeable future.
</item 54>

<item 55>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
And he’s not hopeful we’ll see home prices drop anytime soon: “Absent any negative shocks to the broader economy that would either boost excess supply of homes
on the market or fuel an uptick in unemployment, we continue to expect home prices to rise at a slow pace.” In fact, he predicts we’ll see home prices rise by
1.8% by the end of the year, with a 3.5% increase by the end of 2024.

Looking at more recent housing inventory data makes the issue even more stark. Between September 2018 and September 2023, the average number of homes on the
market dropped a whopping 60% to fewer than 700,000 active listings, according to Realtor.com.

“Unlike the turn of the millennium, house prices today are rising alongside mortgage rates, primarily due to low inventory,” Sam Khater, Freddie Mac’s chief
economist, said in a statement released Sept. 29.
</item 55>

<item 56>
The shipping industry’s boom-and-bust cycle is so severe carriers face going from bumper profits to losing money in the space of a few years: 'I'm certainly
concerned' (Fortune, 2023-11-12T17:51:37+00:00)
Companies including clothing brand Under Armour Inc. and furniture maker Lovesac Co. cited relief from lower ocean shipping expenses over the past quarter.

With inflation eating away at their paychecks, consumers are being cautious about spending and seeking cheaper ways to have parcels delivered.

“Our own data tells us that the wider economic picture may be having an influence on the services our customers opt for, with many looking for more
cost-effective shipping options,” said Karen Reddington, president of FedEx Express Europe. “We expect external business conditions to be challenging in the
near term, and there remains uncertainty with respect to the timing of demand recovery.”

Read More: Powell Says Fed to Be Careful, Won’t Hesitate to Hike If Needed

For the container carriers, the cost of moving merchandise can’t stay this low indefinitely, because their expenses are heading in the opposite direction.
</item 56>

<item 57>
The best apps and tools for managing your money online (The Verge, 2023-11-12T13:00:00+00:00)
Hi, friends! Welcome to Installer No. 14, your guide to the best and Verge-iest stuff in the world. (If you’re new here, welcome, you’re my favorite, so happy
you’re here, and also, you can read all the old editions at the Installer homepage.)

I also have for you a rundown of the best tools for managing money, new gadgets from Valve and Humane, way too many writing apps, new stuff from ChatGPT, and
much more.

As always, the best part of Installer is your ideas and tips. What are you into right now? What have you been reading / watching / learning / doing that
everyone should know about? Tell me everything: installer@theverge.com. And if you know someone else who might enjoy Installer, forward it to them and tell them
to subscribe here.

Big week in the Installerverse! Let’s get to it.
</item 57>

<item 58>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
But this isn’t like 2008 or 2010 again. It’s time for a 1980s history lesson.

Back to the 1980s future

The housing market today isn’t identical to that of the ‘80s, but it’s pretty close. In a lot of ways, millennials are being forced to follow the housing
journey of their boomer parents as they face a frozen, unaffordable market with rising interest rates, as noted recently by both BofA Research economist Jeseo
Park and by First American chief economist Mark Fleming.

Essentially, millennials are a giant generation all collectively coming of homebuying age nearly simultaneously—just like their boomer parents in the 1980s.
They’re the “biggest share of the “homebuying pie,” as Redfin puts it, purchasing about 60% of homes bought with mortgages during the past few years.
</item 58>

<item 59>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
After a long, hot summer that saw mortgage rates creep ever higher, October has brought an early winter for the housing market. Existing-home sales dropped a
stunning 15% in September on a year-over-year basis to a seasonally adjusted annual rate of 3.96 million transactions, according to the National Association of
Realtors (NAR). That’s the lowest figure in 13 years, since 2010, when the world economy, and particularly the U.S. housing market, were struggling to pull out
of the Great Financial Crisis.

Contributing factors to the continuous decline in home transactions include surging mortgage rates (which just hit 8% this week—a record in the 21st century),
low inventory levels, and home prices that refuse to stop rising. In other words, there aren’t enough homes to buy, money isn’t cheap anymore, and the ones for
sale are too rich for most homebuyers’ blood.
</item 59>

<item 60>
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
</item 60>

<item 61>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
Gifts for film and TV buffs

Regardless of whether you’re shopping for a movie buff or an avid sports fan, there are a number of subscriptions on the market that’ll grant your giftee access
to a wide range of content. Below are some of the most popular, as well as a few catered toward anime diehards, horror lovers, and those looking for something
more niche.

The Disney Bundle (monthly subscription, with ads) $ 15 The Disney Bundle caters to all kinds of passions as it grants ad-supported access to Disney Plus, Hulu,
and ESPN Plus for $14.99 a month. That way, sports fans can watch all the latest games, while the rest of the family streams Percy Jackson and the Olympians,
Ahsoka, and The Bear. $15 at Disney

Gifts for the gamers

If you’re not sure which type of games your giftee prefers, you can gift them an Xbox, PlayStation, or Nintendo subscription.
</item 61>

<item 62>
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
</item 62>

<item 63>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
Nintendo Switch lovers, meanwhile, might enjoy an annual subscription to Nintendo Switch Online , which starts at $19.99 a year (Amazon, Best Buy, and
GameStop). The membership lets giftees play with more than 70 retro games released during the NES, SNES, and original Game Boy eras. They can also play online
with friends and access cloud saves.

, which starts at $19.99 a year (Amazon, Best Buy, and GameStop). The membership lets giftees play with more than 70 retro games released during the NES, SNES,
and original Game Boy eras. They can also play online with friends and access cloud saves. If you’re willing to fork out extra money, you can also buy an annual
Nintendo Switch Online + Expansion Pack subscription . In addition to offering all the same benefits as the Switch Online membership, it also grants access to
Game Boy Advance, Sega Genesis, and Nintendo 64 games, as well as DLC content for some titles.
</item 63>

<item 64>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
The Santa bribe turns the focus of Christmas into "selfish gain on the part of your children," Keeley added.

"Love your kids enough to learn how to manage their behavior appropriately and teach them self-discipline."

"It’s also lazy parenting, which never pays off in the long run," she said. "Being a mom is a professional career and deserves to be treated accordingly."

She added, "Love your kids enough to learn how to manage their behavior appropriately and teach them self-discipline."

3. Becoming a ‘holiday martyr’

It’s important to keep the holidays well-rounded, Keeley said.

5 HOLIDAY DECORATING MISTAKES THAT COULD REDUCE YOUR CHEER THIS YEAR

She called out "martyr moms" who are at their "peak performance" during Christmastime, putting everyone else before themselves.

"They wear themselves out making sure everyone else has the perfect Christmas, but never put themselves on the receiving end," she said.
</item 64>

<item 65>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
Fortunately, the internet is filled with travel-oriented gifts, ranging from the obvious — like airline gift cards — to national park passes.

A GetYourGuide gift card provides an easy way to take advantage of guided tours and fun tourist attractions at various destinations around the world, allowing
your loved one to explore volcanoes like Italy’s Mount Etna or swim with sharks in Cape Verde.

provides an easy way to take advantage of guided tours and fun tourist attractions at various destinations around the world, allowing your loved one to explore
volcanoes like Italy’s Mount Etna or swim with sharks in Cape Verde. A Pack & Go gift card (Amazon and Target) covers almost everything your giftee will need
while on vacation. They can use it on a Southwest Airlines flight, buy gas from Chevron or Texaco if they’re on a road trip, and even book an Airbnb or place a
Grubhub order.

(Amazon and Target) covers almost everything your giftee will need while on vacation.
</item 65>

<item 66>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
(Amazon and Target) covers almost everything your giftee will need while on vacation. They can use it on a Southwest Airlines flight, buy gas from Chevron or
Texaco if they’re on a road trip, and even book an Airbnb or place a Grubhub order. For aspiring polyglots, a Rosetta Stone membership will make it easy to
learn everything from French to Arabic to Japanese from the comfort of their home. Along with lessons, memberships come with helpful extras, including speech
recognition tech to get the accent just right. At the moment, Rosetta Stone is offering up to 50 percent off of its plans, with the 12-month tier for learning
just one language starting at $126 ($42 off).

will make it easy to learn everything from French to Arabic to Japanese from the comfort of their home. Along with lessons, memberships come with helpful
extras, including speech recognition tech to get the accent just right.
</item 66>

<item 67>
The best Apple Black Friday deals on iPads, AirPods, Apple Watches and more (Engadget, 2023-11-24T09:30:32+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

There are still tons of Apple Black Friday deals you can shop for online; these are our favorites.

Most of Apple’s current-generation devices went on sale for Black Friday — and most of those deals look like they’ll carry over through the weekend and into
Cyber Monday. That means if you were otherwise occupied on Friday, you can still save on iPads, Apple Watches, MacBooks, AirPods and more. Some of the prices
are new lows and many others match the record prices we’ve seen previously. We’ve tested and reviewed nearly every device below, so check out the links to our
full reviews and guides if you need help deciding which deals to snag. These are the best Apple Black Friday deals you can still get today from retailers like
Amazon, Target, Best Buy and B&H Photo.
</item 67>

<item 68>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
They can use it to buy whatever they need for the kitchen, as well as to take online cooking classes where live instructors help students make everything from
chicken piccata to tiramisu. Classes start at $29 per household, with each taking between 90 and 120 minutes a pop. For those with a sweet tooth, Goldbelly’s
monthly ice cream subscription allows them to enjoy up to six pints of ice cream or 24 ice cream sandwiches a month, all of which are sourced from small
creameries all over the country. It’s a pricey subscription, however, with a three-month plan going for an eye-watering $299.85.

allows them to enjoy up to six pints of ice cream or 24 ice cream sandwiches a month, all of which are sourced from small creameries all over the country. It’s
a pricey subscription, however, with a three-month plan going for an eye-watering $299.85.
</item 68>

<item 69>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
Subscribers also get discounts on select digital books.

is perfect. For $5.99 a month, they’ll be able to enjoy over 45,000 comics and graphic novels, not to mention more than 2,400 manga titles from their phone or
tablet. Subscribers also get discounts on select digital books. You can also gift a Book of the Month membership , which currently starts at $59.99 for a
three-book plan. The company curates a small selection of five to seven bestsellers and classics for members every month, making it easy to quickly choose
something to read.

, which currently starts at $59.99 for a three-book plan. The company curates a small selection of five to seven bestsellers and classics for members every
month, making it easy to quickly choose something to read. If your giftee prefers to pick up books from brick-and-mortar bookstores, a gift card to Barnes &
Noble (Barnes & Noble, Amazon, or Best Buy) might be a good idea.
</item 69>

<item 70>
Here are over 40 last-minute gifts you can still nab before Christmas (The Verge, 2023-12-24T19:15:58+00:00)
At the moment, Rosetta Stone is offering up to 50 percent off of its plans, with the 12-month tier for learning just one language starting at $126 ($42 off).
Finally, a gift card to a rideshare service like Uber (Amazon, Best Buy, Target) or Lyft (Amazon, Best Buy, Target) will come in handy when they need a ride to
the airport. It will also come in handy if their New Year’s Eve plans include popping a few celebratory shots.

Gifts for health and wellness fans

For health and wellness enthusiasts, many services offer a wealth of streamable fitness classes to help them get fit at home. Other gifts can help your giftee
practice self-care and lighten their load with meditation or even massages. Below, we’ve listed out a range of options that’ll help your giftee take care of
both their body and mind.
</item 70>

<item 71>
The Verge’s 2023 holiday gift guide for kids (The Verge, 2023-12-18T23:14:07+00:00)
Price: $116.10+ The Shop at The Broad MoMA Design Store

Arteza Gel Ink Pens (60-pack) Some kids will always complain about school and homework, but the right set of pens can make both just a teeny-tiny bit more
enjoyable. These long-lasting gel pens from Arteza are great for writing, drawing, coloring, or just doodling. The metal-tipped pens come in a wide range of
colors, too, with metallic, pastel, and glitter finishes.

Price: $17.84+ Amazon Target Arteza

We’re Not Really Strangers Kids Edition Having deeper conversations with children is rarely easy, but the kids edition of We’re Not Really Strangers is an
enjoyable exercise that gets both adults and children closer to the same wavelength. The thoughtful prompts are likely to elicit unexpected answers from both
you and your kids, and the rainbow-colored pencils are a playful touch.
</item 71>

<item 72>
Fast fashion retailer ASOS struggles to engage consumers as company losses balloon (The Independent - Life and Style, 2023-11-01T15:36:20+00:00)
“So curated collections and recommendations based on user input may be one of the changes they eventually look at.”

ASOS, which originally stood for As Seen On Screen, initially started trading with the tagline “Buy what you see on film and TV” as it exclusively sold
imitations of clothing worn by celebrities. That business model quickly expanded away from the world of replica products: ASOS now offers 850 affordable to
mid-range high-street brands and ships to 196 countries.

But the company is also now tackling a debt issue, with net debt including leases now at £648.5m, up from £533m the year before. Analysts predict that the
online fashion site may need to raise quick cash soon – potentially through the rumoured sale of its Topshop brand, which ASOS acquired when the Arcadia Group
went bust in 2021 and closed Topshop’s bricks and mortar stores.
</item 72>

<item 73>
The best Black Friday headphone deals from Sony, Apple, Bose, Beats and more (Engadget, 2023-11-24T14:30:56+00:00)
Engadget Editor Jon Turi found this pair to deliver more low-end depth than most of its peers, however. It can also last up to 10 hours per charge.

Your Black Friday Shopping Guide: See all of Yahoo’s Black Friday coverage, here. Follow Engadget for Black Friday tech deals. Learn about Black Friday trends
on In The Know. Hear from Autoblog’s experts on the best Black Friday deals for your car, garage, and home, and find Black Friday sales to shop on AOL,
handpicked just for you.
</item 73>

<item 74>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
USPS 2023 Holiday Shipping Deadlines Final day to ship Ships via this service December 20th First-Class Mail December 20th USPS Ground Advantage December 21st
Priority Mail December 21st Priority Mail Express

Thursday, December 21st, is the final day to send packages (and have them arrive by Christmas Eve) through USPS via its Priority Mail Express service. You can
also get packages delivered by December 23rd via Priority Mail. First-Class Mail packages and those you intend to send via USPS Ground Advantage, meanwhile,
must be sent by Wednesday, December 20th, to arrive in time.

Illustration by Alex Castro / The Verge

If you subscribe to Amazon Prime and live in an eligible ZIP code in the US, you can get Same-Day Delivery on certain items through December 24th. You’ll need
to meet the “minimum threshold” of eligible items in your cart, however, which can vary by location (see here for more details). In LA, for example, the
threshold is over $25.
</item 74>

<item 75>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
Hopefully, it’ll help you avoid some holiday stress, if at all possible.

UPS 2023 Holiday Shipping Deadlines Final day to ship Ships via this service December 19th UPS 3 Day Select December 20th UPS 2nd Day Air December 21st UPS Next
Day Air

With UPS, you have until Thursday, December 21st, to ship via Next Day Air and have your package arrive by Friday, December 22nd. Packages shipped via Next Day
Air after Thursday won’t arrive until after Christmas. Wednesday, December 20th, is the last day to ship via 2nd Day Air if you want your package to arrive by
Friday, December 22nd. You can opt for UPS’s 3 Day Select option until Tuesday, December 19th, to have it arrive by the 22nd. Before then, you can select the
normal pickup and delivery service to get your packages where they need to be on time.
</item 75>

<item 76>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
As a bellwether for the rest of the holiday period — traditionally the biggest period for retail sales in the year — the flat sales speak to another tough year
for online retailers.

Adobe Analytics is predicting $37.2 billion in online spend for the full five days, up just 5.4% year on year and accounting for 16.8% of all holiday spend. And
Black Friday will see $9.6 billion in sales, up roughly the same, 5.7%, versus figures last year. (The published figures in 2022 were $9.13 billion.) Salesforce
has not provided forecasts.

For some context on today’s Thanksgiving figures, last year’s published figures from Adobe were $5.29 billion, which actually represents an increase of just
under 4%. (It’s likely Adobe Analytics readjusted its final figures for last year, which is why we see a higher percentage of growth.)
</item 76>

<item 77>
The best Black Friday deals you can get for under $25 (The Verge, 2023-11-23T17:58:09+00:00)
Every Black Friday and Cyber Monday, there’s always a ton of hype about the discounts on big-ticket items — you know, the deals on the OLED TVs, game consoles,
and tablets of the world. But it never fails that many of the ultra-cheap items make for some of the most clutch deals, namely because spending very little on a
small accessory you end up using all the time can be the most gratifying.

This year, throughout the annual shopping weekend, we’re once again keeping an eye out for the best tech deals on those little gadgets and gizmos under $25.
We’ve also got roundups for those working with $50 or $100 budgets, but the items below are pretty much as cheap as you can get in the tech world without simply
buying absolute crap that’s destined to end up as e-waste.

Some of these helpful budget deals include gems like fully compatible MagSafe chargers for the latest iPhones, a colorful smart speaker, Apple’s handy AirTag
trackers, and some staff favorites.
</item 77>

<item 78>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Salesforce’s calculations of average order value were equally modest. Globally, average orders were up just 2% to $103 per “basket” while in the U.S. they were
up a paltry 1% to $119.

Mobile devices had a standout year: Adobe said that some $3.3 billion was spent over mobile devices on Thanksgiving, up 14% and an all-time record for the day.

Salesforce added that online traffic in general to e-commerce sites, which includes browsing, was up too, but again only in single digits of 4% globally and 6%
in the U.S.

The two are actively tracking sales for today, Black Friday, and for the whole of the “Cyber Weekend” — which not only kicks off holiday shopping but has a
particularly strong showing in online sales due to people traveling to be with family.
</item 78>

<item 79>
The housing winter arrives early as existing home sales crash to Great Recession levels and the long-feared ‘deep freeze’ sets in (Fortune,
2023-10-20T20:30:29+00:00)
And while we haven’t seen home sales activity levels this low since the 2000s, the housing market more closely mimics that of the ’80s, according to a report
published this week by Fortune 500 financial services company First American.

“Today’s housing market isn’t anything like the housing market of the mid-2000s,” First American’s Fleming wrote in a Tuesday report titled “1980s Déjà Vu for
the Housing Market.” “The housing market today is not overbuilt, nor is it driven by loose lending standards, subprime mortgages, or homeowners who are highly
leveraged.”

While some economic factors are stronger today than they were during the GFC, housing affordability is undeniably worse.

“While housing and more generally consumer fundamentals are in a much stronger position today, affordability for the incremental buyer is worse than it was at
the peak in 2006 before the crash,” Roger Ashworth, a managing director at Goldman Sachs, wrote in a credit strategy research paper released last week.
</item 79>

<item 80>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
Plugs like these are a simple way to add some smart capabilities to any home, letting you turn on lights with just your voice, set automated schedules and
routines that are triggered by other activities. These would make a great stocking stuffer for anyone you know who's curious about smart home connectivity but
hasn't yet taken the plunge.

Xbox gift card

You can pick up a $50 Xbox gift card for $45 right now at Amazon and Target, a digital item that would make a good gift for a gamer who’s otherwise hard to shop
for. This card can be used on a number of things, including game downloads, in-game purchases, membership costs like Xbox Game Pass and more. If you or a loved
one is loyal to a different console, there are other digital gift cards on sale, too. A $70 PlayStation Plus gift card is going for $63 right now, while a
12-month Nintendo Switch Family plan is slightly discounted from $35 to $32.
</item 80>

<item 81>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
However, the arrival date of your package may vary depending on whether Walmart or a third-party seller is shipping your item.

The retailer offers in-store pickup through December 24th, depending on availability, as well as same-day delivery on December 24th. Same-day pickup is
available on orders made by 12PM local time.

You can also have the package delivered to you in less than two hours on Christmas Eve if you order by 12PM local time and pay a $10 Express Delivery fee. Note
that Walmart Plus members also get one Express Delivery free during the month of December.

Illustration by Kristen Radtke

Both same-day delivery and in-store pickup are available to you if you place an order by 6PM local time on Christmas Eve. It helps to choose your desired
location as “My store” in the upper-left corner of Target’s website, which will prompt it to automatically indicate if the item you’re purchasing is available
for pickup nearby.
</item 81>

<item 82>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Park and her team noted that there are no “noticeable signs” of excess housing development today like there were back then, and households aren’t nearly as
burdened by mortgage debt. Household mortgage debt represented roughly 65% of U.S. consumers’ disposable income in the second quarter, compared to 100% before
the GFC. And the ratio of Americans’ mortgage debt to their real estate assets—also called loan-to-value—was just 27% in the second quarter, compared to over
40% in 2008 and roughly 50% in 2010, Bank of America’s data shows.

New legislation was also enacted since the GFC to help prevent worst-case scenarios in the housing market. One of the most obvious effects of these new laws is
that there are far fewer risky adjustable-rate mortgages today.
</item 82>

<item 83>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
Roughly 60% of Gen Zers say they are stressed about money this year more than last year, according to a Bankrate survey from July. It’s no surprise either—85%
of Gen Zers say that they couldn’t afford one month’s expenses if they lost their job today.

And since young people are typically affected by inflation the most, as they are the most likely to work part-time or low-paying jobs, this economic climate may
have left Gen Z with permanent “psychological scars,” one expert says.

“How can young people build careers or wealth if they don’t have jobs, and prices of goods and services continue to increase?” Dayo Abinusawa, founder of
London’s Awa Business School and a former lecturer at Cambridge University’s Judge Business School, previously told Fortune.

One Fidelity survey backs this argument: 45% of 18- to 35-year-olds “don’t see a point in saving until things return to normal.”
</item 83>

<item 84>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
These parents, said Keeley, "subconsciously compensate for a lack of self-worth and validate their role as a [parent] by creating too much busyness in their
life."

She said that "all that results from a mom who runs herself into the ground is just that: a mom who runs herself into the ground."

DELICIOUS CUT-OUT SUGAR COOKIES FOR CHRISTMAS: TRY THE SIMPLE RECIPE

She added, "If this is you, take a breath, put away the to-do list and ask yourself, ‘What would I have to believe in order to have a relaxed and joyful
Christmas?’ Maybe we can use that as a guide to sit back and relax."

4. Griping like the Grinch

Parents should be sure not to be "grinchy" around their kids amid holiday stress, Keeley said.

Children up to the age of 9 process information egocentrically, she said — which means if there’s a problem, they see themselves as the cause.
</item 84>

<item 85>
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
</item 85>

<item 86>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
And while I’ve long been a proponent of MPOWERD’s solar-powered Luci lanterns, I recently picked up the company’s like-minded string lights for car camping and
overnight jaunts in the backcountry when I don’t mind toting a little extra weight with me.

The 18-foot string is certainly not the brightest you can buy — it packs a series of 100-lumen LEDs, whereas your average headlamp might offer 400 — but it can
swap between six different colors and features a 2,000mAh battery for when your phone needs some emergency juice. Best of all, you can charge the lights via USB
or solar, meaning you can spend less time worrying about how to keep them going and more time taking in the vibes.

Memory card holder and reader

Becca Farsace, senior producer

As a video person who is constantly on the move, there is nothing better than the consolidation of gear — especially when it pertains to dongles.
</item 86>

<item 87>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
EXPERIENCES VS. GIFTS

Another great way to save money is to skip material gifts and instead, grant your time. In many cases, your family would rather spend quality time with you.

Alev recommended planning low-cost activities such as a hike with your family, or attending a free event in your city. Other options can be a holiday movie
marathon with your friends, a free museum with your parents or a volunteering day with your family.

USE CREDIT RESPONSIBLY

Since Plauché is already paying her student loan debt every month, she is avoiding using her credit card this holiday season to avoid carrying more debt.

If using your credit card is your only option, it’s important that you use it wisely.

If you struggle with overspending during the holidays, Alev also recommends setting a budget for each of your credit cards.
</item 87>

<item 88>
Adobe: Thanksgiving US online sales nudge up to $5.6B; Salesforce: $31.7B spent globally (TechCrunch, 2023-11-24T15:43:51+00:00)
Thanksgiving Thursday, when stores in the U.S. are closed and many are spending time off work, has become the de facto start of the holiday shopping season both
for those looking to get bargains online and for online retailers to kick off holiday sales deals to meet that demand. But if there is a message from this
year’s Thanksgiving sales, it is that consumers are holding steady, but don’t hold your breath for a growth boom.

Adobe Analytics said that people in the U.S. on Thursday spent $5.6 billion online, which it calculates at an increase of just 5.5% on last year.

Salesforce, which also calculates sales based on data collected by its Commerce Cloud division, noted that globally, online sales reached $31.7 billion with its
U.S. tally at $7.5 billion — both up only 1%. Salesforce’s calculations of average order value were equally modest.
</item 88>

<item 89>
26-year-old with $145,000 in student debt can't afford Christmas. She's giving 'affordable gifts or combining gifts with my older sister' (Fortune,
2023-12-23T15:13:12+00:00)
BUDGET

Alev is a big proponent of the 50-30-20 budgeting method, where 50% of your income is allocated for necessities like food and rent, 30% for things you want, and
20% for savings and debt repayment.

To find space in your budget for gifts, Alev recommends that you plan ahead as much as possible and only take money from the 30% portion of your budget.

If the 50-30-20 is not the right format for you, there are plenty of other budgeting methods that you can use. Whether is recording your spending on an Excel
spreadsheet, using an app or writing it down on a notebook, keeping a budget is most effective when you do it in a format that works for you.

If you are looking for a budget template, Microsoft Office offers a holiday spending-focused version.
</item 89>

<item 90>
The new American Dream is to get MrBeast to pay off your debt (TechCrunch, 2023-12-21T14:00:20+00:00)
And, to be extremely on the nose, MrBeast created his very own Squid Game challenge, where 456 people competed for $456,000. Then, Netflix also created a
reality competition show based on the dystopian thriller.

Of course, the contestants on Netflix or in MrBeast videos are participating voluntarily and are not in mortal danger. Still, we’re living in a country where
the total amount of student debt has almost tripled in the last 15 years to over $1.77 trillion dollars. And according to a study from Kaiser Health News and
NPR, 41% of American adults have some form of medical debt.

If getting sick can render you bankrupt, why not live inside a grocery store for 45 days if it could eliminate your financial burdens? The American Dream is no
longer the promise that anyone can get rich if they just work hard enough.
</item 90>

<item 91>
Some of our favorite devices are on sale for Black Friday. (The Verge, 2023-11-24T15:42:21+00:00)
It was still worth it, though.)

Fuzzy cat bed

Elizabeth Lopatto, senior reporter

Jeeves loves to be in the office with me while I work. (I assume she’s supervising.) Anyway, to make her cozier, I got her this fuzzy cat bed. At first, she was
afraid of it — she’s extremely shy around new people and objects — but now, a year later, it’s her favorite place to sit that isn’t my lap. Usually, she spends
the afternoon snuggled up in it.

Distracting dog toy

Kaitlin Hatton, audience manager

Trudee, otherwise known as My Boss, is notoriously insistent on being the focus of my attention. At any given moment, she has 20 toys strewn about my apartment,
but she will not play with them unless I’m watching her — like literally, no distractions, sitting within feet of her, just staring at her chew on her toys for
hours on end.

Don’t get me wrong.
</item 91>

<item 92>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
CHRISTMAS TREE SHOPPING? HERE'S HOW TO SELECT THE PERFECT ONE FOR YOU

"Aside from the time I asked my husband to turn our bathroom into a Hallmark movie and he actually pulled it off, unrealistic expectations around Christmas
usually just set up moms for inevitable disappointment," Keeley said.

"Beautiful family Christmas memories can be made without the drive-in ‘Grinch’ movie, the tacky light tour and the neighborhood Christmas cookie bake-off all in
the same weekend."

Instead, parents should "take it easy and leave plenty of room to relax," said Keeley.

2. Using Christmas as a bribe

While it’s easy for parents to use a message of "be good or else" as a parenting tactic before Santa Claus comes down the chimney, Keeley said this approach
lowers parenting standards.

"I refer to this tactic as ‘stooping to Santa,’" she said. "Don’t lower your parenting standards to bribery, especially around a season and a character that is
supposed to symbolize peace and goodwill on Earth."
</item 92>

<item 93>
The 26 best Amazon Prime Day deals you can still shop for today (Engadget, 2023-10-12T15:44:16+00:00)
Engadget has been testing and reviewing consumer tech since 2004. Our stories may include affiliate links; if you buy something through a link, we may earn a
commission. Read more about how we evaluate products.

Amazon's October Prime Day sale officially ended on Wednesday, but a few of the deals are still live as we head into the weekend. If you were hoping to check
off a few more names from your holiday gift list or buy something for yourself at a discount, see what remains on sale below. We combed through our coverage
over the past few days to find the best of what's left. While Amazon has returned most of their own devices, like Echos and Kindles, back to their full prices,
a handful of products from other brands like Google, Samsung and Anker, have maintained the lows they hit on Prime Day. Here are the best deals from Prime Big
Deal Days that you can still get right now.
</item 93>

<item 94>
The best apps and tools for managing your money online (The Verge, 2023-11-12T13:00:00+00:00)
Once again, you came through! I got a ton of emails, texts, and mentions with good ideas for apps and systems for making this all work. Also, a lot of people
who are very cranky with Mint, and trust me, I feel you on that.

One note before we get into it: you should, of course, be careful about where you manage your money and financial information and to whom you give access to any
of that data. Many of the apps we’re about to talk about are popular and highly regarded, but you should always be cautious with this stuff. My credit score
also once got dinged because I had signed up for too many personal finance apps — true story — so there are lots of reasons to be thoughtful here.

Cool? Cool. Let’s run through some of your ideas and findings:

YNAB is the big winner. YNAB, which stands for You Need A Budget, was the overwhelming most popular recommendation I got this week.
</item 94>

<item 95>
Buckle up for more ‘turbulence’ in the housing market, BofA says—it’s a housing recession, 1980s style (Fortune, 2023-10-06T20:05:26+00:00)
Housing market pessimists have been sounding the alarm for years about a pending crash in the U.S. residential real estate market. Even before the Federal
Reserve began hiking interest rates to fight inflation last year, pushing mortgage rates to a 23-year high this month, years of surging home prices left some
experts warning that the housing market was a massive bubble ready to pop.

However, despite a widely bearish outlook in the industry, most real estate veterans have avoided arguing that home prices will drop like they did during the
2008 crash that kicked off the Global Financial Crisis (GFC). And so far, that’s been a wise decision.

It took until June 2022 for national home prices to peak, even as mortgage rates soared and purchase applications plummeted amid the Fed’s first interest rate
hikes. And while prices fell more than 5% from that peak by January of this year, they returned to a record high in July.
</item 95>

<item 96>
The 5 biggest mistakes parents make at Christmastime, according to a parenting expert (FOX News - Lifestyle, 2023-12-09T19:46:55+00:00)
Pressures are high for parents and caregivers around the holidays.

That means there’s room for error, according to parenting expert Hannah Keeley, who offered up some of the biggest mistakes parents tend to make at
Christmastime.

In an email she sent to Fox News Digital, Virginia-based Keeley identified the top five holiday blunders that parents often make.

HOME FOR THE HOLIDAYS? HERE'S HOW TO AVOID STRESS AROUND FAMILY MEMBERS, ACCORDING TO EXPERTS

Here are the mistakes Keeley thinks parents should try to avoid.

Diving right in …

1. Creating unrealistic expectations

Parents who make a "must-do" Christmas list that’s a "mile long" at the "first sniff of peppermint mochas" are perhaps setting themselves up for failure — and
Keeley said she empathized with this instinct.

Doing the most could lead to a season of disappointment, she warned.

CHRISTMAS TREE SHOPPING?
</item 96>

<item 97>
The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more (Engadget, 2023-11-19T15:13:23+00:00)
The the magnetic clip mount and long-lasting, fast-charging battery are also reasons to buy.

Masterclass Black Friday deal

Masterclass’ Black Friday deal gives you two memberships for the price of one, starting at $120 per year. That provides two individual memberships, while you
can pay a total of $180 for access for two devices simultaneously, content downloads and a Family membership that covers six devices. Masterclass could be a
good gift option for people who love to learn or collect hobbies as it provides access to hundreds of hours of video of masters of their craft explaining how
they do their work. Some classes have exercises you can complete as well, making it feel more like a real class. This deal will be available until the end of
the day on Black Friday proper.

Headspace Black Friday deal

Headspace Headspace $35 $70 Save $35 Headspace has knocked 50 percent off its memberships for Black Friday, so you can pay only $35 to get one year of access.
</item 97>

<item 98>
Here are the shipping deadlines to get your holiday gifts on time (The Verge, 2023-12-18T19:26:44+00:00)
You can also get packages within an hour on Christmas Eve (if you order before 4PM local time) with a service called Shipt. An annual Shipt subscription
provides unlimited deliveries on orders above $35 for $99 a year, though an annual membership is currently going for half off. Alternatively, you can pay for
same-day shipping with Shipt on a per-order basis via Target’s website. Shipping fees may vary depending on your location and the contents of the order. Here
are answers to some frequently asked questions that might help.

Target RedCard credit card holders receive free two-day shipping on many items from the retailer’s website.

Many (but not all) orders over $35 are eligible for free two-day shipping, even if you don’t have a Target RedCard. It’s determined by the item’s location and
your ZIP code.

Target stores will be open on Christmas Eve, though it’s wise to get pickup orders and same-day shipping orders in by December 23rd to be on the safe side.
</item 98>

<item 99>
Fast fashion retailer ASOS struggles to engage consumers as company losses balloon (The Independent - Life and Style, 2023-11-01T15:36:20+00:00)
“Once large online retailers get to a certain size, it can become laborious for customers to scour their whole catalogue,” Smith explains. “The online fashion
landscape is evolving now to a point that customers are looking for a more tailored service from websites.”

With the rise of shopping on social media, too, young people are increasingly buying individual items from retailers on Instagram and TikTok or instead shopping
directly from the brands that ASOS stocks, such as Pull and Bear, Stüssy and Bershka, which all have their own websites.

ASOS distribution centre near Barnsley, South Yorkshire. (PA)

“ASOS’s busy 20-something [target] market doesn’t necessarily have the time to sift through dozens of item pages,” adds Smith. “So curated collections and
recommendations based on user input may be one of the changes they eventually look at.”
</item 99>

<item 100>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
However, it likely won’t be the last opportunity for you to save. As someone who’s been covering deals for The Verge and Polygon for years, I’m shocked by how
often I see these low prices, often spoken of as if they’re a flash in the pan, extending through much of December. That said, it’s more likely that price cuts
on products you want can go by unnoticed unless you’re tracking deals yourself.

Track deals yourself

We love to tell you about great deals. But there’s a limit to the kind of products that we write about, and the ones that we see. If you’re not comfortable
leaving it up to the chance that we’re going to tell you about price cuts on the products you really want, you have the power to track deals with relative ease.

If you only shop on Amazon, CamelCamelCamel remains one of the best tools for tracking price changes and history for products.
</item 100>
=== END ITEM id=b325c467186b2b69 ===

=== ITEM id=b0ec2eadd60b2f21 ===
Gold sentences and facts:
Sentence 0: In blog posts and press materials, Google touted Gemini’s superior architecture and capabilities, claiming that the model meets or exceeds the performance of other leading gen AI models like OpenAI’s GPT-4.
  Fact 0.0: Google touted Gemini's superior architecture and capabilities in blog posts and press materials.
  Fact 0.1: Google claimed that Gemini meets or exceeds the performance of other leading gen AI models.
  Fact 0.2: OpenAI's GPT-4 is one of the other leading gen AI models Google named.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: The woman admits that she may have misinterpreted the evidence.
  Fact 1.1: The woman maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:

=== END ITEM id=b0ec2eadd60b2f21 ===
