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

This file has 3 items.

=== ITEM id=083a97cf9a5f626c ===
Gold sentences and facts:
Sentence 0: “Even with those tools, some of the businesses selling on Amazon might still choose to set prices that aren’t competitive,” Zapolsky said.
  Fact 0.0: Zapolsky said that even with those tools, some businesses selling on Amazon might still choose to set prices that are not competitive.
Sentence 1: That means eBay will give you an eBay coupon for the difference in price between what you bought the item for and what the competitor is selling it for, plus an additional 10 percent of that price difference.
  Fact 1.0: eBay will give the buyer an eBay coupon for the difference in price between what the buyer paid for an item and what a competitor is selling it for.
  Fact 1.1: The eBay coupon also includes an additional 10 percent of that price difference.

Retrieved items:
<item 1>
Discogs’ vibrant vinyl community is shattering (The Verge, 2023-10-07T14:00:00+00:00)
He’s been selling on Discogs since 2009; since the recent changes, he’s lowered his prices to offset the higher shipping costs and was compelled to institute an
order minimum — a major shift for a marketplace that had done considerable business in selling records under $10 and even under $5.

Discogs attributed the need to raise fees to its “significant investments in recent years to ensure compliance with various regulatory programs, including tax
support and privacy protection.” The company said the change would allow it to “continue to devote resources to maintaining the Discogs Marketplace and develop
better tools for collecting, selling, and enjoying music.”

Many sellers who spoke with The Verge speculated, in line with the viral thread, that the company was trying to pump up its valuation for a potential sale. All
of them, though, had the sense that Discogs was trying to increase its profit margins without necessarily offering any improvements to its product in return.

Discogs’ marketplace page showing copies of Olivia Rodrigo’s Guts for sale. The prices are generally higher than buying the album elsewhere.
</item 1>

<item 2>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
However, it likely won’t be the last opportunity for you to save. As someone who’s been covering deals for The Verge and Polygon for years, I’m shocked by how
often I see these low prices, often spoken of as if they’re a flash in the pan, extending through much of December. That said, it’s more likely that price cuts
on products you want can go by unnoticed unless you’re tracking deals yourself.

Track deals yourself

We love to tell you about great deals. But there’s a limit to the kind of products that we write about, and the ones that we see. If you’re not comfortable
leaving it up to the chance that we’re going to tell you about price cuts on the products you really want, you have the power to track deals with relative ease.

If you only shop on Amazon, CamelCamelCamel remains one of the best tools for tracking price changes and history for products.
</item 2>

<item 3>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
It also doesn’t apply to products that require that you log in to see the price, “Marketplace” items from third-party sellers, or lower-priced items purchased
from other Target stores.

Best Buy’s Price Match Guarantee states that a product could be eligible for a price reduction if it’s new, identical to the competitor’s product, immediately
available at a qualifying competitor’s store or on its site, and not shown on Best Buy’s exclusion list (which includes “items for sale the Friday before
Thanksgiving Day through the Monday after Thanksgiving”).

To get the difference, you have to contact Best Buy via its on-site chat and be able to show the ad or website with a lower price point. Alternatively, you can
call 1-888-237-8289 or visit any Best Buy or Pacific Sales store.

Best Buy will also, upon request, match an in-store or online price that was lowered during the return and exchange period, which typically lasts up to 15 days.
</item 3>

<item 4>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Seeking Alpha's Quant Rating system turned bullish on the stock just a few days before the earnings by upgrading its view to Strong Buy from Hold. Meanwhile,
Wall Street analysts maintain their optimism about the stock with a Strong Buy rating.

After more than a year of rigorous examination by antitrust investigators across the world, Microsoft just completed its $69B acquisition of Activision
Blizzard, the largest deal in video game history. Its benefits, however, will not be reflected in the company's financial report.

Consensus EPS Estimates: $2.65

Consensus Revenue Estimates: $54.54B

Earnings Insight: Microsoft has outperformed EPS forecasts in 7 of the last 8 quarters, and revenue in 6 of those quarters.

Also reporting: General Electric Company (NYSE:GE), Verizon Communications (VZ), Visa Inc. (V), The Coca-Cola Company (KO), 3M Company (MMM), General Motors
Company (GM), Raytheon Technologies Corporation (RTX), Teladoc Health, Inc.
</item 4>

<item 5>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
The EU has reached this pass of laying out formal concerns after conducting what it describes as a “wide-ranging investigation to understand the market and the
potential impact of the deal”, which it says included analysing internal documents provided by Amazon and iRobot and gathering views from market participants
such as suppliers of RVCs and other smart home devices, as well as from providers of online sales channels.

It also says it closely cooperated with other competition authorities during both the initial investigation and the in-depth investigation, adding that it will
continue to do so during the remainder of the in-depth investigation. Although the U.K.’s competition authority already cleared the deal back in June.

Amazon was contacted for a response to the EU’s statement of objections. A company spokesperson sent us this statement in which it claims iRobot faces “intense
competition” from other RVC suppliers:

We continue to work through the process with the European Commission and are focused on addressing its questions and any identified concerns at this stage.
</item 5>

<item 6>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
Today’s announcement by the EU, following several months of in-depth examination looking at how the transaction might impact competition, lays out formal
concerns for Amazon to respond to. The EU’s objections focus on foreclosure risks.

The EU says its concerns focus on whether the deal will give Amazon the ability and incentive to foreclose iRobot’s rivals by deploying strategies aimed at
preventing competitors from selling RVCs on Amazon’s marketplace and/or making it harder for them to do so — such as by delisting rival products; reducing their
visibility in both non-paid (i.e. organic) and paid results (i.e. ads) displayed on its marketplace; limiting their access to certain widgets it operates (such
as the “other products you may like” feature) or “certain commercially-attractive product labels” (e.g. ‘Amazon’s choice’ or ‘Works With Alexa’); and/or by
directly or indirectly raising the costs of iRobot’s rivals to advertise and sell their RVCs on its marketplace.
</item 6>

<item 7>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
If you find a lower price from a qualifying competitor on an eBay Deal item that you bought within 48 hours — or on another product with the “Best Price
Guarantee” symbol attached to it — you’ll get “110 percent” of the difference. That means eBay will give you an eBay coupon for the difference in price between
what you bought the item for and what the competitor is selling it for, plus an additional 10 percent of that price difference. Note, however, that these
coupons will expire within 30 days of being issued.

The products must be new, unopened, and identical to the competitor’s item. You have to contact eBay Customer Service, and if they verify your claims are
legitimate, you will receive the coupon and be able to use it toward your next eBay purchase.

GameStop does not mention an official price-matching policy on its website. Last year, however, GameStop tweeted that the store will price-match items sold and
shipped by Amazon as long as they’re available online or in-store.
</item 7>

<item 8>
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
</item 8>

<item 9>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
Regardless, you need to notify Newegg of the lower price within 14 days of purchase. You’ll then be issued a credit you can use on the Newegg website, though
you’ll only have 90 days to do so.

Newegg’s price-match guarantee is not valid during major holiday sales events, including Black Friday, Cyber Monday, and Green Monday.
</item 9>

<item 10>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“We can tailor our ads to hit exactly the consumers we want,” Moller said. “It’s a perfect marriage of a transaction, and that’s one of the beauties of what
their marketplace offers.” Needham said he feels he would have been more supportive of the case if it were filed a few years ago, pre-pandemic. At that time, he
said, "I would have felt, yes Amazon is a monopoly… But actually after Covid, into 2023, ecommerce has had a lot of big changes.” He added, "The competition is
just not what it was in 2019." Competitors like Shopify and Walmart are increasingly viable alternatives for many categories of sellers, Needham said, not to
mention rapidly growing Chinese e=commerce companies like Temu. As a result, Needham said he’s seen some significant changes from Amazon. Among those is a
greater ability for Amazon sellers to communicate with buyers, offering select customers certain promotions.
</item 10>

<item 11>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Fred Ruckel

Nearly three years later, Ruckel’s experience hits at the core of a sweeping antitrust lawsuit filed last week by the Federal Trade Commission against Amazon.
The agency accused Amazon of wielding its monopoly power to squeeze merchants and thwart rivals. For consumers, that’s led to artificially inflated prices and a
degraded shopping experience, the agency alleges. In the 172-page suit, the FTC said Amazon relies on an “anti-discounting strategy” and a “massive web-crawling
apparatus that constantly tracks online prices” to stifle competition. The agency said Amazon punishes third-party sellers who offer cheaper products elsewhere
by threatening to disqualify them from appearing in the buy box if it detects a lower price. Losing the buy box is an “existential threat” to sellers’
businesses, the complaint alleges. The end result of these tactics, the FTC argues, is elevated prices across the web.
</item 11>

<item 12>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Shopify, for example, gives sellers much more control over how they communicate with customers, Needham said, adding that although Amazon still controls the
communication process, at least there is one. "I wish it was a clear-cut case,” Needham said. “I have a vested interest in the marketplace doing really well, as
a seller and as a service provider. And... this case, it doesn’t make the marketplace better for sellers."

Concerns over Amazon pricing policies, fees

Many sellers have zeroed in on Amazon’s pricing policies and rising fees as rightful areas of concern in the FTC's lawsuit. Molson Hart, whose company Viahart
sells toys on Amazon, has been a longtime critic of Amazon’s pricing policies. Hart complained of how Amazon’s seller fees impact pricing in a 2019 Medium post
and later that year testified about his experience before a House committee.
</item 12>

<item 13>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
At the time the EU’s then competition chief, Margrethe Vestager, trumpeted the outcome as setting what she dubbed “new rules for how Amazon operates its
business in Europe” — arguing the settlement meant the ecommerce giant could “no longer abuse its dual role” as marketplace operator and own brand retailer; and
further claiming the outcome would ensure “competing independent retailers and carriers as well as consumers will benefit from these changes opening up new
opportunities and choice.”

Given the commitments the EU already extracted from Amazon after those earlier investigations it’s interesting to see the bloc remains concerned over levers the
ecommerce behemoth still has its disposal to potentially influence competitive outcomes on its marketplace vis-à-vis iRobot vs rival robot vacs.

Since then Amazon has also been named a gatekeeper under the EU’s shiny new ex ante competition regulation, the Digital Markets Act — which lists its
marketplace as a core platform service. This designation means Amazon must comply with a series of up-front obligations, which include restrictions on self
preferencing.
</item 13>

<item 14>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Meta's cost-cutting actions and a rebound in ad income boosted the stock's performance this year, which has climbed by about 150%. The stock has been rated as a
Strong Buy by sell-side analysts and Seeking Alpha's Quant Rating system.

Recently, Alphabet's unit Google (GOOG) (GOOGL), Meta Platforms (META), and Apple (AAPL) have received approval from U.S. regulators to deploy the new set of
mobile AR/VR devices, including goggles and in-car connections.

Consensus EPS Estimates: $3.60

Consensus Revenue Estimates: $33.43B

Earnings Insight: Meta has topped EPS expectations in 4 of the past 8 quarters, beating revenue estimates in 5 of those reports.

Boeing (BA)

Boeing (BA) is preparing to report Q3 results after the closing bell on Wednesday.
</item 14>

<item 15>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Amazon in recent years made the forum anonymous, but users must have a seller account in order to post. Another post included a screenshot of a message Amazon
sent to sellers the day after the FTC filed its complaint, which said, “As your partners, we know that this news may generate questions for you and our business
together. This lawsuit does not change anything about our relationship with you or how we operate today.” One user called it “BS verbiage,” adding, “Businesses
that sell in their store are indeed customers. And which of us has gotten good customer service?” Another user described their experience in the last 12 months
of selling on Amazon as “being up all night at an effing casino but I'm stuck, the drugs are starting to wear off, but I'm trying to break even on the mortgage
payment I'm using to play. That's how it is selling on Amazon right now to me.” The seller went on to describe the experience as a “race to the bottom.”
</item 15>

<item 16>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
Lastly, there are always exceptions to the rules. If you’re looking for a deal on a product and you know it’s available for a lower price elsewhere (or it was
previously offered at the same retailer), it never hurts to call or chat with a customer service rep and ask if they’ll match the price. After all, what do you
have to lose?

If Best Buy offers a better discount than Target on the latest Kindle Paperwhite, Target will refund you the difference. Photo by Chaim Gartenberg / The Verge

Amazon doesn’t offer price matching. Full stop. You can contact customer support, but even if you’re an Amazon Prime customer, they’re not likely going to help
you out.

That said, Amazon does offer a pretty liberal return policy, meaning you can often return an item you purchased from Amazon if it’s available for less elsewhere
(or at Amazon). Then, you can just repurchase the product wherever it’s cheapest.
</item 16>

<item 17>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
European Union regulators have sent Amazon a formal statement of objections setting out competition concerns over its proposed acquisition of robot vacuum
maker, iRobot.

The step does not confirm the EU will seek to block the deal but it shows antitrust regulators do remain concerned. It could therefore push Amazon to offer
remedies to EU authorities to settle the investigation and avoid the risk of the acquisition being blocked.

Amazon has already lowered its acquisition price for iRobot — which was originally an all-cash offer of $1.7 billion back in August 2022 — by 15% owing to
increased debt taken on by its target while regulatory snags delayed closing the deal at the original purchase price.

The bloc has been closely scrutinizing the Amazon-iRobot deal since July when the EU announced its in-depth probe. The Commission said at the time it was
worried the transaction would allow Amazon to restrict competition in the market for robot vacuum cleaners (RVCs) and strengthen its position as online
marketplace provider in a number of ways.
</item 17>

<item 18>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
It also doesn’t apply to membership stores like Costco and Sam’s Club.

If you find a comparable product at a lower price during the return period, you can email HP at myhpsales@hp.com. This only applies to branded HP products and
doesn’t apply to products purchased during certain promotional events. This includes Green Monday and products on sale during the week of Thanksgiving and the
week after.

Lenovo matches pricing on comparable PCs from Lenovo, HP, or Dell with the same key specs as long as they’re sold through a list of recognized retailers. The
comparable PC must also be in stock and readily available from both Lenovo and the competing retailer, and you have to contact a Lenovo sales rep in order to
facilitate the match.

That said, you can’t take advantage of coupons or special discounts while receiving a price match, and the policy doesn’t apply to membership stores like
Costco. The policy also doesn’t apply to “Marketplace” items where third parties fulfill the order, so double-check that you bought the product directly from
the retailer.
</item 18>

<item 19>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 19>

<item 20>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
PayPal’s fees are, however, driven by the costs imposed upon them by the dominant card schemes, Visa, Mastercard and American Express. So, we can expect PayPal
will respond to this suit by arguing that it is as much of a price taker as the merchants themselves, and that the terms imposed upon it by the card schemes
drive some of its anti-steering rules.”

PayPal did not respond to requests for comment. — Mary Ann

Bolt CEO gets frank about SEC probe

Bolt Financial CEO Maju Kuruvilla told me in an interview that the one-click checkout company is putting a recent U.S. Securities and Exchange Commission probe
behind it and is moving on.

“It obviously is a very, very involved process, but we’re really happy to put that behind us,” Kuruvilla told TechCrunch.
</item 20>

<item 21>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
“Amazon may have the ability to foreclose iRobot’s rivals because Amazon’s online marketplace is a particularly important channel to sell RVCs in France,
Germany, Italy, and Spain,” the Commission writes in a press release. “RVC customers in these countries particularly rely on Amazon both in terms of product
discovery as well as for their final purchasing decision.”

The Commission is also worried Amazon may have the incentive to foreclose iRobot’s rivals because it may be economically profitable for it to do so. “The merged
entity would likely gain more from additional sales of iRobot RVCs, than it would lose from fewer sales of iRobot’s rivals and other related products on Amazon.
Such gains include benefits from additional data gathered from iRobot’s users,” it suggests.

Were Amazon to deploy such foreclosure strategies the Commission’s view at this stage of the probe is it could restrict competition in the market for RVCs,
leading to “higher prices, lower quality, and less innovation for consumers”.
</item 21>

<item 22>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
Walmart doesn’t price-match the prices of its competitors, though Walmart’s online store offers price-matching on items purchased from Walmart stores if the
item is in stock at Walmart.com.

However, the policy doesn’t apply to those who live in Alaska, Hawaii, or Puerto Rico. Additionally, it doesn’t apply to special event prices, like those seen
during Black Friday. Walmart also does not retroactively price-match items bought from Walmart.com that have dropped in price after the time of purchase. The
retailer additionally doesn’t price-match items purchased from Walmart Marketplace retailers or third-party sellers.

Apple doesn’t offer a price-matching policy, though you can price-match on devices like the 15-inch MacBook Air — assuming it’s on sale — elsewhere. Photo by
Amelia Holowaty Krales / The Verge

While we’ve seen some claims that Apple will price-match up to 10 percent for products bought at certain major retailers, there is no official policy on the
company’s website.
</item 22>

<item 23>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
The seller went on to describe the experience as a “race to the bottom.” “It's long overdue," another commenter wrote. "When they close me down, I'm applying
for a job with the FTC.” Still, others commented that the FTC’s complaint is misguided. “Selling on Amazon is a life-changing opportunity and the amount of
sellers that throw stones at the platform is astounding,” one user wrote.

Seller skepticism

Even sellers who may be sympathetic to the idea of regulating Amazon have concerns, specifically that the FTC's highlighted issues aren't necessarily ones that
would make the seller and consumer experience better. Scott Needham, who sells on Amazon and runs a product-finder tool for other Amazon sellers, said he was
“surprised by some of the points that the FTC selected.”

watch now

“I have over the years been very critical of Amazon,” Needham told CNBC. “I’ve lost a lot of sleep because of some of the things that they have done.
</item 23>

<item 24>
Discogs’ vibrant vinyl community is shattering (The Verge, 2023-10-07T14:00:00+00:00)
But those alternatives, being considerably larger and more mainstream, offer a much broader base of potential buyers as well as a more solid infrastructure and
support system.

“eBay has much more of a user base, so for the little bit of extra cost it’s a no brainer,” says Martin, who says that, for him, eBay’s fees are usually around
1 percent higher than Discogs’. “It’s probably double [the business] I do on Discogs, and that’s only grown since they raised the fees.” He sells primarily new
vinyl and uses Amazon, Walmart, eBay, and Discogs, along with his own website, apocalypsevinyl.com. With the new fees and the competitiveness of the Discogs
market, the platform is becoming less and less useful as a selling channel.

“They’re under the impression that they’re the only game in town,” says the Connecticut seller of Discogs. “The fees were relatively low, but now that they’re
higher, there doesn’t seem to be a reason to use that anymore.”
</item 24>

<item 25>
Amazon’s iRobot purchase sucks up formal competition concerns in EU (TechCrunch, 2023-11-27T23:29:10+00:00)
iRobot, which faces intense competition from other vacuum cleaner suppliers, offers practical and inventive products. We believe Amazon can offer a company like
iRobot the resources to accelerate innovation and invest in critical features while lowering prices for consumers.

Late last year the EU settled with Amazon over two earlier competition probes — one delving into antitrust concerns about Amazon’s use of merchant data to
bolster its own retail business; and a second one looking at how it operated the “Buy Box” and its Prime loyalty program.

In those cases Amazon agreed a series of commitments to end the probes — including saying it would stop using non-public data from its marketplace sellers and
boost transparency for merchants; and pledging to treat all sellers equally, regardless of whether they pay for its logistics services.
</item 25>

<item 26>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“But I’m not sure if this would actually change that.” Scott Moller, an Amazon seller and co-founder of an agency that helps merchants run their storefronts,
said the e-commerce giant has removed some of the challenges that used to be part of running an online business. With FBA, he said, he can ship an item into one
of Amazon’s warehouses for $7.49 per package, while shipping it himself through a traditional carrier would cost him about $12. “I don’t have to have my own
warehouse,” said Moller, who sells grilling accessories on Amazon under the brand Grill Sergeant. “I can use their staff, their storage, and I can instantly
also take the data of advertising, so I can target ads.” He also disputed the FTC’s claim that Amazon has become littered with ads in search results, causing
shoppers to wade through potentially less-relevant products of lesser quality. “We can tailor our ads to hit exactly the consumers we want,” Moller said.
</item 26>

<item 27>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
The stock gets a Hold rating from Seeking Alpha's Quant Rating system, a more cautious view than the consensus Buy rating from Wall Street analysts.

"Cleveland-Cliffs is well-positioned to perform well over time, thanks to its strong position when it comes to supplying automobile companies and due to energy
cost advantages compared to competitors from markets such as Europe," says investing group leader Jonathan Weber.

Consensus EPS Estimates: $0.44

Consensus Revenue Estimates: $5.53B

Earnings Insight: The company has beaten EPS expectations in 3 of the past 8 quarters and revenue in 5 of those reports.

Also reporting: Logitech International S.A. (LOGI), Koninklijke Philips N.V. (PHG), Packaging Corporation of America (PKG) and more.

Tuesday, October 24

Alphabet (GOOG) (GOOGL)

Alphabet (GOOG) (GOOGL) will be announcing its Q3 results during Tuesday’s post-market session.
</item 27>

<item 28>
Discogs’ vibrant vinyl community is shattering (The Verge, 2023-10-07T14:00:00+00:00)
The prices are generally higher than buying the album elsewhere.

“It just seems like they’re actively trying to stop sales,” says the Connecticut seller. “You can raise your fees, but maybe you could do some promotions,
coupon codes, sales — something that offsets the shift. Sellers can do it on their own, but that’s going to require them to lower their prices — it’s going to
be a race to the bottom. If you were trying to ruin a sales forum, this is how you’d do it.”

Discogs did have a sale in late August, but it featured just 11 of the site’s largest stores. “When I first saw it, I thought, maybe they’ll be randomly
promoting stores or the best products,” says Martin. “I don’t know a big sale that most people are not part of that you promote to further depress our prices is
the right direction.”

The website itself is a frequent source of complaints, as is the lack of support.
</item 28>

<item 29>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
A worker sorts out parcels in the outbound dock at the Amazon fulfillment center in Eastvale, California, on Aug. 31, 2021. Watchara Phomicinda | MediaNews
Group | The Riverside Press-Enterprise via Getty Images

It was late in the day on Oct. 27, 2021, when Fred Ruckel received the dreaded automated email from Amazon . Amazon’s software had detected that Ruckel’s
popular cat toy, called the Ripple Rug, was being sold somewhere else for a cheaper price. His product would no longer be shown in Amazon’s all-important buy
box, an area of the listing where shoppers click “Add to Cart.” Ruckel is the sole seller of the Ripple Rug on Amazon, so the move all but ensured his product
would disappear from the website, costing him thousands of dollars per day.
</item 29>

<item 30>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
Listen to TC+ editor Alex Wilhelm and I talk more about how there’s plenty of market share to go around in Friday’s episode of the Equity podcast below. — Mary
Ann

PayPal is being sued for alleged “Draconian” payment policies

On October 5, Mary Ann broke the news that PayPal has been hit with a class action lawsuit by consumers represented by law firm Hagens Berman alleging that the
fintech giant’s anti-steering rules stifle competition against lower-cost payment platforms such as Stripe and Shopify. Specifically, according to an
investigation conducted by the firm’s consumer rights attorneys, PayPal has subjected consumers to excess charges when purchasing from online merchants that
accept PayPal or Venmo.

Someone commented on social media that they didn’t see the problem with what PayPal is doing and questioned whether or not Visa and Mastercard do the same
thing. So I went back to the attorneys who filed the lawsuit, who noted that the “anti-steering rules” are not the first of their kind.
</item 30>

<item 31>
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
</item 31>

<item 32>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
Consensus EPS Estimates: $-2.53

Consensus Revenue Estimates: $18.37B

Earnings Insight: Boeing has beaten EPS expectations just once in the past 8 quarters and revenue just twice in that span.

Also reporting: International Business Machines (IBM), QuantumScape Corporation (QS), ServiceNow (NOW), General Dynamics (GD), Automatic Data Processing (ADP),
Baker Hughes Company (BKR), KLA Corporation (KLAC) and more.

Thursday, October 26

Amazon (AMZN)

Ecommerce and cloud giant Amazon (AMZN) is due to release its Q3 earnings on Thursday after the market closes. The quarter will focus on Amazon's cost-cutting
initiatives, supply chain management, and AI product deployment as potential positive factors for attracting investment in equities amidst current market
environments.

Wall Street analysts have a consensus Strong Buy rating, but Seeking Alpha Quant Ratings assesses the company as a Hold. On a YTD basis, Amazon shares have
expanded by almost 50% in value.
</item 32>

<item 33>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
In October 2019, Epic internally said it might pursue an aggressive pursuit model with the Epic Games Store, where it would aggressively pay more for exclusive
games and really pull in more and more users. Are you in the aggressive pursuit model?

No, we’re in a different model, which is called the “Ridiculously aggressive pursuit model.” If it’s a 10-slide deck, that’s on slide 11.

Did you get a bingo on your Verge bingo card?

Yeah, I think it was almost a complete shutout. I think there was only one cell in the entire board that was missing. I was really impressed — the thing for
that bingo card that really impressed me was “lick the cookie.” Do you remember? You may not even be old enough, but “lick the cookie” was in the 1999 US v.
Microsoft antitrust trial.
</item 33>

<item 34>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
Hart said Amazon sales comprise about 90% of his business, meaning any hit those sales take on Amazon has a considerable impact. He recalled “24 anxious hours”
in September 2022 when a third-party seller of his popular construction toy Brain Flakes listed the toy for a lower price on Target than it was offered on
Amazon.

Molson Hart, CEO of Viahart, an educational toy company that sells on Amazon. Courtesy: Molson Hart
</item 34>

<item 35>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
The search-engine giant has outperformed the broader market index with a 55% growth rate this year, surpassing the S&P 500's 12%.

Following the tech behemoth's upbeat Q2 results, investment firm Bank of America said it is expecting "favorable" headlines, due in part to strength in YouTube,
search, and artificial intelligence.

Just a few days before the earnings report, Seeking Alpha's Quant Rating system changed its view from Strong Buy to Hold, while Wall Street analysts maintained
a bullish view on the stock, giving it a Strong Buy rating.

Consensus EPS Estimates: $1.45

Consensus Revenue Estimates: $75.73B

Earnings Insight: Google has topped EPS expectations in 4 of the past 8 quarters, beating revenue estimates in 5 of those reports.

Microsoft (MSFT)

Joining Alphabet on Tuesday, Microsoft (MSFT) is also set to report its Q4 performance after the closing bell.
</item 35>

<item 36>
Discogs’ vibrant vinyl community is shattering (The Verge, 2023-10-07T14:00:00+00:00)
To make up for the lost revenue, Discogs suggested sellers use a tool it had created to raise the prices of all of their inventory by a percentage; another
Discogs email to sellers suggested they offer free shipping to avoid the fee, without accounting for the fact that the seller would then be either covering that
cost out of pocket or integrating it into the price of the record — which would, of course, result in the same amount of money going to Discogs. Essentially,
sellers were told to raise their prices and / or offer free shipping — two options that threaten their bottom lines. “Their communication, too — it’s like, ‘I
said what I said, and we’re done,’” says Martin. “Well, you’re really not, because we all have to live with this and so do you.”

The tension between Discogs’ old-internet charm and its attempts at growth came to a head earlier this summer around a since-deleted viral Twitter thread by
artist and label head Mike Simonetti lamenting “the fall of discogs.”
</item 36>

<item 37>
Discogs’ vibrant vinyl community is shattering (The Verge, 2023-10-07T14:00:00+00:00)
According to the sellers who spoke with The Verge, bugs abound: the European seller, for example, had just been dealing with an issue with the platform’s refund
button. “Discogs said it was PayPal’s fault, and PayPal said it was Discogs’ fault,” he says. “It caused stress for the buyers, and so I had to do direct
refunds — which meant I was refunding not just what I made but Discogs’ and PayPal’s commissions as well, effectively losing money on the refund.” Melkisethian,
speaking a month later, had just noticed a shift in the way shipments are processed that required manually entering information in steps that used to be
automated.

Besides the baseline functionality of the site, there are other improvements that could bring Discogs closer in line with its competitors. “There are other
seller tools and seller initiatives that we’ve been asking for for years that have never been done — like any kind of tie-in with Google, any kind of
integration with social media, the kinds of things basically other platform has,” says Martin.
</item 37>

<item 38>
Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more (Seeking Alpha, 2023-10-22T12:13:00+00:00)
On a YTD basis, Amazon shares have expanded by almost 50% in value.

"AMZN's cost-cutting initiatives and margin improvements make it a solid buying opportunity, despite concerns such as the FTC lawsuit and a weakening economy."
writes SA Investing Group Leader Danie Jones with a bullish stance on the stock. Meanwhile, fellow Investing Group Leader Sensor Unlimited expects the stock to
continue being range-bound in the near term due to the key challenges on its operating margins, competition in key growth areas, and valuation risks.

Consensus EPS Estimates: $0.60

Consensus Revenue Estimates: $141.621B

Earnings Insight: Amazon has beaten EPS and revenue expectations in half of the past 8 quarters.

Also reporting: Kenvue (KVUE), Intel (INTC), Ford Motor Company (F), Altria Group (MO), Merck & Co.
</item 38>

<item 39>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Below is a list of product(s) in your catalog that are not currently eligible to be the Featured Offer because they are not priced competitively compared to
prices for those products from retailers outside Amazon,” according to the email, which was viewed by CNBC. Unbeknownst to him, Chewy was running a discount
promotion, and dropped the price of his product by a few dollars to $39.99 – less than the $43 offer on Amazon. The algorithm had flagged it as a lower offer,
even though the item on Chewy cost $48.54 after shipping and taxes. Ruckel had to make a choice: Lower the price on Amazon or ask Chewy to raise the price of
his product. He opted for the latter.

Fred Ruckel's company Snuggly Cat makes Ripple Rug, an interactive play mat for cats.
</item 39>

<item 40>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
Then, you can just repurchase the product wherever it’s cheapest.

As of October 22nd, Target will match the price of any item purchased online or in-store from Target that goes on sale through December 24th. However, the
current price-matching policy doesn’t apply to products you may have bought from one Target store and found for less at another.

Target will also match the price of items purchased from Amazon, Best Buy, and a select group of competitors if you ask for it at the time of or within 14 days
of your purchase. You must bring the original ad or listing, whether digital or in print, along with your receipt into the store. Alternatively, you can call
for online purchases at 1-800-591-3869 or chat online via contactus.target.com.

Be mindful, however, that Target’s policy excludes doorbusters and competitors’ lightning sales, like those that Amazon offers.
</item 40>

<item 41>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
If you bought the product from a Costco warehouse location, visit the membership counter at the store where you made the purchase. Costco doesn’t price-match
items bought from Costco warehouses with Costco.com purchases.

Dell will price-match identical products from Dell, Amazon, Best Buy, Walmart, Target, HP, Apple, Lenovo, and a number of other retailers. You can call
1-877-275-3355 or chat online to run a price check, but you’ll need to provide a link to the competitor’s product.

You can also contact the company about a price difference within 30 days of purchase. However, there is a long list of restrictions, and none of this is
available between Thanksgiving Day and the Monday following Thanksgiving (or on Green Monday).

As a general rule, eBay doesn’t match the prices of purchases made during the weeks of Thanksgiving and Cyber Monday. It does, however, offer a generous
price-matching policy for items bought before or after that time period.
</item 41>

<item 42>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
The end result of these tactics, the FTC argues, is elevated prices across the web. The company steadily hikes the fees it charges sellers and prevents them
from discounting on other sites, so sellers often inflate their prices off of Amazon, creating an “artificial price floor everywhere,” according to the
complaint. The FTC is seeking to hold Amazon liable for allegedly violating anti-monopoly law, though it has not yet outlined the specific remedies it believes
would best resolve its concerns. In antitrust cases, remedies are often determined only after a court finds the defendant liable. In a blog post, Amazon general
counsel David Zapolsky said third-party sellers set their own prices on the marketplace. The company also invests in tools to help sellers offer “competitive
prices,” he said. “Even with those tools, some of the businesses selling on Amazon might still choose to set prices that aren’t competitive,” Zapolsky said.
</item 42>

<item 43>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“I’ve lost a lot of sleep because of some of the things that they have done. And the issues that they brought up, while they are interesting, they haven’t
created me a lot of pain.” Needham said he was particularly puzzled by the inclusion of the claims that Amazon is coercive in the way it encourages sellers to
use its fulfillment service, known as Fulfillment by Amazon, or FBA. Needham said many sellers “love FBA” because of its compelling value in terms of the price
and promise to deliver two-day shipping. For many, using FBA doesn’t feel like a requirement, but they believe using it will make their businesses "easier and
more effective." "I think that the power that Amazon wields over sellers is considerable and absolutely worth looking into,” Needham said. “But I’m not sure if
this would actually change that.”
</item 43>

<item 44>
How to tackle Black Friday shopping like a boss (Polygon, 2023-11-12T15:00:00+00:00)
Amazon does not price match, which is annoying, but it’s tolerable since it often has the lowest prices of all retailers anyway. Amazon only protects the price
on pre-orders that have its “Pre-order Price Guarantee,” which ensures that you’ll pay the lowest amount a product sells for before it’s launched.

Not only does Best Buy match prices from “qualified competitors,” but its site says that most products bought between now and Dec. 30 can be price-matched
through Jan. 13, 2024.

Walmart’s policy says it won’t price match with competitors, nor does it price match for products purchased online that later decrease in price.

Target says that purchases made between Oct. 22 and Dec. 24 can be price-adjusted if the cost lowers in the span of time, and if you show proof of purchase. You
may be eligible for a price match on a purchase, but the rules for submitting documentation vary depending on whether you’re shopping online or in store. It’s
all explained here.
</item 44>

<item 45>
Here’s how Rainforest, a budding Stripe rival, aims to win over software companies (TechCrunch, 2023-10-08T14:16:01+00:00)
They said: “Visa and MasterCard once imposed similar anti-steering rules on merchants accepting their cards but, after the Justice Department sued the networks
for antitrust violations, they agreed in 2010 to eliminate their anti-steering rules as part of the settlement. With payments transitioning into the digital
realm, PayPal has simply ripped a page right from the Visa and MasterCard [sic] playbook.”

Meanwhile, Patrick McGahan, a partner at Scott+Scott who focuses on antitrust litigation, had an interesting take that actually involves both of those card
giants. He told TechCrunch that the case illustrates that “the tensions between merchants and the payment system providers are not over, and that litigation
regarding this key cost faced by merchants is likely to continue for some time.”

He added: “Platform companies that operate as gatekeepers in their market, such as PayPal, will continue to be the subject of antitrust litigation as a result
of the significant fees they charge.
</item 45>

<item 46>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
Apple does say that if it reduces its price on an Apple-branded product within 14 days from when you received your item, you can visit an Apple Store or contact
the Apple Contact Center at 1-800-676-2775 to request a refund or get credit for the price difference. However, that doesn’t apply to special sales like Black
Friday or Cyber Monday.

B&H says that it might be able to honor a current lower price or match the price of a competitor, but it does so on a case-by-case basis. You have to contact
its Customer Service team using the on-site chat, email, or by calling 800-606-6969 / 212-444-6615.

Costco does not price-match with competitors. However, according to the website, it will issue a credit for the difference between the price a member has paid
and a promotional price if you bought the item online at Costco and requested the credit up to 30 days before the change in price.
</item 46>

<item 47>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
Nothing is more frustrating than buying a new pair of headphones, an OLED TV, or a backpack just to find out that you could have gotten it for a lot cheaper
somewhere else. In order to keep customers happy — and prevent them from going elsewhere — many retailers offer price-matching policies where they promise to
match a lower price you’ve found elsewhere. That’s information that could come in handy now that Black Friday and Cyber Monday are just around the corner.

What follows are the price-matching policies for a variety of major retailers. There are a few things to make note of here. First, all price-matching policies
have a number of rules attached to them. We’ve summarized some of the rules here, but you’d be advised to go to the links that we’ve provided and read the
policy carefully. Second, and logically enough, retailers that serve as marketplaces for other sellers, such as Costco, don’t have price-matching policies.

Lastly, there are always exceptions to the rules.
</item 47>

<item 48>
Amazon sellers sound off on the FTC's 'long-overdue' antitrust case (Cnbc | World Business News Leader, 2023-10-06T21:31:00+00:00)
“Just like any store owner who wouldn’t want to promote a bad deal to their customers, we don’t highlight or promote offers that are not competitively priced.”
Zapolsky argued the FTC’s lawsuit could force it to stop highlighting low prices, “a perverse result that would be directly opposed to the goals of antitrust
law.”

“Long overdue” lawsuit

On Amazon’s own forum for merchants, called Seller Central, several users cheered on the FTC and said they hoped it would result in changes to the company’s
business practices. Amazon’s tense relationship with merchants has been well-chronicled over the years, with sellers expressing a range of grievances over
issues like rising fees, an arcane suspensions process, and heightened competition on the marketplace from all sides, including the e-commerce giant. “I think
it’s great, Amazon deserves it,” one person commented, adding, “More should be coming on the way.”
</item 48>

<item 49>
The inside story of Dave Clark's tumultuous last days at Flexport (Cnbc | World Business News Leader, 2023-10-02T17:46:00+00:00)
Shopify had acquired Deliverr in May 2022 for $2.1 billion. But the e-commerce software company was getting hammered by Wall Street as its Covid pandemic pop
faded. By January 2023, CEO Tobias Lutke knew he needed to get rid of Deliverr. Around that time, Lutke first approached Petersen to float the possibility of a
deal, according to a person familiar with the matter. Petersen told Clark he should engage with Shopify's team, according to a person with direct knowledge of
the negotiations. Initial talks fell apart, but resumed when Flexport executives learned that Shopify was about to execute deep cost cuts and was eager to sell
Deliverr. Clark and Petersen flew to Miami to meet with Shopify's leadership.
</item 49>

<item 50>
Know the price-matching policies for Best Buy, Target, Walmart, and others (The Verge, 2023-10-25T22:03:19+00:00)
Although GameStop only mentioned Amazon in the tweet, others have reported that the store will price-match items purchased from Target and elsewhere.

Google will match online pricing from select retailers — specifically Amazon, Best Buy, Walmart, B&H Photo, and Target — if a request is made within the
standard return period. It doesn’t, however, match preorder pricing from other retailers or carrier deals. The product also must be identical, right down to the
color and configuration, and must be in stock. Plus, you need to provide a public link to the promo page or product listing to verify the rival’s price.

You may also get the difference if a product purchased from the Google Store lowers in price within the standard return period. To get the price match or
correction, contact Google.

The Home Depot will match the price of identical, in-stock products from other retailers, except for membership-based wholesalers.
</item 50>
=== END ITEM id=083a97cf9a5f626c ===

=== ITEM id=4fd4df3b342b04ef ===
Gold sentences and facts:
Sentence 0: But sportsbooks are also gathering this information, and they begin to tighten the NBA Rookie of the Year lines in response.
  Fact 0.0: Sportsbooks are also gathering this information.
  Fact 0.1: Sportsbooks begin to tighten the NBA Rookie of the Year lines in response.
Sentence 1: Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like an injury or roster move.
  Fact 1.0: Every sportsbook jumps on board when a line moves for a legitimate reason.
  Fact 1.1: Every sportsbook jumps on board when odds change for a legitimate reason.
  Fact 1.2: An injury or a roster move is an example of a legitimate reason.

Retrieved items:
<item 1>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
That means checking the posted lines and odds for the same bets at different sportsbooks.

You can do this for moneylines, point spreads, totals bets, prop bets, and more. Finding the best line or better odds on any bet is like finding money.

You can shop around using multiple sportsbook apps on your phone or live odds feeds like the one above. Pick a side, then use the drop-down menus and tabs to
find the odds and lines for that bet.

Finally, bet when you see an opportunity at better odds or an improved line compared to the competition.

Line shopping for futures

People tend to bet futures at their favorite online sportsbook. However, shopping for better odds on any future bet is worth your time. Most game lines are
similar at different sportsbooks, but the odds in futures markets can vary significantly.

Sportsbooks sometimes have higher liabilities for specific teams and offer better odds on others to lay that off. Timing is everything.

Get it right, and you can easily take advantage of changes in different futures markets due to liability.
</item 1>

<item 2>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Pay attention to player and team news: Oddsmakers use various information to develop their lines for games and players, and you should, too. Keep tabs on player
news and injuries, when top players are being rested, how they performed in recent games, and more. You can find value in player and team props by gathering
information from multiple sources and tracking how players and their teammates perform. For example, if one of the starting guards is questionable for a game,
you might want to consider the other guard for increased points or assists since they’ll likely handle a heavier load.

Keep an eye on usage and minutes: Even slight increases to a player’s average minutes can make a difference when betting player props, especially later in the
season when aging veteran players tend to rest more. Keep yourself informed on usage and minutes trends for the players you’re considering betting on, and use
them to your advantage.
</item 2>

<item 3>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Cons:

There is a reduced opportunity to find value.

The odds may not be as favorable.

Keep in mind that it’s always important to take into consideration your risk tolerance while you weigh the potential rewards. No matter when you decide to make
your wager, always gather as much information as possible before risking your hard-earned money.

Who are the past NBA ROTY winners?

For context, here is a list of the National Basketball Association Rookie of the Year winners since 2010.
</item 3>

<item 4>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Caesars Sportsbook: Caesars knows its way around sports betting when you use its app and online sportsbook. It offers favorable odds for almost every bet type,
including NBA ROTY betting, and offers a nice variety of deposit and withdrawal options. Additionally, Caesars has a competitive welcome bonus for new players
and runs NBA betting promos for existing players.

When is the best time for NBA ROTY odds?

The earlier you bet on NBA ROTY, the better the odds. But you’ll be doing so without much information that could be key in making an educated decision.

Conversely, if you wait for the data you need for an informed bet, the odds will tighten up, and you will see less of a return on your wager.

When the lines are first released for NBA ROTY honors, the season hasn’t even started yet, so there are no statistics, trends, or player news. The pre-season
odds are so favorable because even the oddsmakers don’t have the data to craft more specific lines.
</item 4>

<item 5>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Once the season has started and you get to see how the various rookies are performing, you can gather the relevant data. But sportsbooks are also gathering this
information, and they begin to tighten the NBA Rookie of the Year lines in response.

So, while you can better guess which player will hoist the Wilt Chamberlain Trophy, the odds will be tighter, and your potential return on your wager will be
reduced.

Pros and cons of betting early on the NBA Rookie of the Year:

Pros:

You have an opportunity to find better value.

The betting odds are fresh and haven't been adjusted.

Cons:

There is a much higher risk as a player's performance, health, and team dynamic can change throughout the season.

You’ll find limited information to make an informed bet.

Pros and cons of waiting to bet on the NBA Rookie of the Year:

Pros:

There is a lot more information available to make an informed bet.

You don’t risk betting on a player not remotely in contention.

Cons:

There is a reduced opportunity to find value.
</item 5>

<item 6>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks will move the line or change the odds if the public bets heavily on one side.

Every sportsbook’s goal is to even out the betting on both sides of every bet. That allows sportsbooks to earn the profit built into the odds and lines without
worrying about which side wins.

The betting is rarely the same at every sportsbook. While one sportsbook may be forced to move a line or change the odds because of betting, another won’t.
That’s when you’ll spot a difference between sportsbooks. This difference may allow you to book a bet at improved odds or a more favorable line.

The lines and odds may even be different right from the jump. Either way, the differences are why line shopping always makes sense. Shop around, and you can
find opportunities to lay less, win more, and improve your chances.
</item 6>

<item 7>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
The significant turning point came in 2018 when the US Supreme Court made a pivotal decision. It overturned a longstanding federal law and allowed individual
states the autonomy to determine their own sports betting regulations.

This groundbreaking decision sparked a wave of change. Since that landmark ruling, over 30 states have embraced sports betting, including wagers on the NBA. For
those residing or visiting betting-friendly states, the experience promises fairness and security.

State-regulated online betting platforms are in place, ensuring that bettors receive fair odds. Furthermore, these platforms adhere to strict standards,
ensuring a safe and transparent betting environment for everyone involved.

Related Pages:
</item 7>

<item 8>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Ensure the site has a responsible gaming section.

The overall user experience should be smooth and hassle-free.

Safety and Security:

Top NBA betting sites prioritize users' safety.

Ensure your personal information is secure.

Some sportsbooks excel in ensuring cybersecurity; always go for those with solid reputations.

How to find the best NBA lines

Not all sportsbook betting odds are created equal. If you're willing to put in some legwork, you can shop for the best NBA lines to maximize your potential
return.

Movement in the odds depends on multiple factors, including team news, injuries, public betting, recent play, and even the number of games a team plays in a
row.

Oddsmakers use all the available information to craft their lines, and you can find value in shopping around for the most favorable lines and odds.

You can use our NBA odds page (see link above) to find the best lines for your prospective bet, or you can open accounts at multiple sportsbooks and do the
shopping yourself.
</item 8>

<item 9>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Getting more points or laying fewer on a typical NFL point spread or NBA point spread might lead to winning a bet you’d otherwise lose. Plus, if you find better
odds on any bet, you can potentially drop fewer dollars when you lose and earn more when you win.

Shopping lines will help you consistently find the best prices on your bets. It won't help you pick more winners, but it will improve your chances and lead to
bigger wins and more affordable losses. That’s undoubtedly worth your time.

For example, you might like the Kansas City Chiefs to beat the Cincinnati Bengals in a mid-season NFL game. Say you look around and see that three of four
licensed online sportsbooks in your state have the Chiefs as -3 favorites, but the fourth has them at just -2.5.

Let’s say the Chiefs win by exactly three, a distinct possibility since a single field goal decides most NFL games. Bets at the first three sportsbooks would
push. However, you’d have a winner at the fourth.
</item 9>

<item 10>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
What makes NBA betting special or unique?

Basketball odds are known to fluctuate more than those of other sports. The factors that play into those lines shifting can range from having only a few impact
players per team to the sheer frequency of games.

By following a single team, you can track its momentum swings, which players tend to get on hot or cold streaks, or how well they play at home vs. on the road.
You can use that information to your advantage for both pre-game and live wagers.

Similar categories of moneylines, spreads, totals, and NBA prop bets should appear across both formats.

The NBA is also special because of the number of bets available for every game, ranging from pre-game to live betting and parlays.

When you add up the standard bets, game props, player props, in-game betting, and more, there’s a staggering volume of bets for every NBA game at online
sportsbooks.
</item 10>

<item 11>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
If you need to use a bonus or promo code, enter it at this stage. Once you’ve signed up, make your initial deposit, and you’re ready to start NBA betting.

See more: Best NBA betting promos

What to look for in an NBA betting site

Here are some key factors to consider:

Deposits and Payouts:

Check if funds are available instantly after depositing.

Understand the variety of deposit methods the sportsbook offers.

For payouts, know the withdrawal methods and wait times.

Be aware of any limits or restrictions on withdrawals.

NBA Odds:

Always select betting sites offering the best NBA odds for long-term benefits.

Odds can differ among sportsbooks; getting the best deal can increase your winnings.

For instance, consider using both if one book offers the Boston Celtics at +800 and another at +1500.

User Experience:

Assess how easy it is to navigate the site.

Gauge the efficiency of customer support.

Ensure the site has a responsible gaming section.

The overall user experience should be smooth and hassle-free.
</item 11>

<item 12>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
However, the only way to find these opportunities is shopping lines.

See more: NFL futures bets

Can I get a better line or odds after I’ve already bet?

When you bet, you lock in the odds or line as currently posted. If they change, for better or worse, you’re stuck with the odds from when you wagered.

That’s precisely why you should go line shopping before you bet. Of course, if significant changes occur, you can take more action at the improved odds or more
favorable line.

Is line shopping worth my time?

Because it can provide you the opportunity to lay less, win more, and improve your chances, line shopping is certainly worth your time. Plus, it doesn’t have to
take up all that much of it.

Use the live odds feed above to do a basic survey. You’ll find it takes just seconds to compare lines and odds. Plus, download and install multiple licensed
online sportsbook apps on your phone, and you can do a more thorough shop in just a few minutes.
</item 12>

<item 13>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
However, you’d have a winner at the fourth. If you didn’t shop around, you’d never have found that half-point victory and be stuck with a push instead of a
profit.

What is the Vegas line?

Las Vegas sportsbooks set the standard for lines and odds across the country. If a sportsbook is copying a line, it’s usually copying the Vegas line. Most
online betting sites don’t stray too far from the Vegas line, using it as a foundation for setting all lines in every major sport.

If those lines do shift, it’s usually because of increased betting on one side of a bet. Shop lines locally, and you can take advantage when sportsbooks stray
from the Vegas line.

Why do the lines move?

It’s worth reiterating that sportsbooks want an equal amount of money bet on each side of every bet. When that happens, they earn the small profit built into
the odds without concern about who wins or loses.
</item 13>

<item 14>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
We urge you to read the requirements for any available promotions and offers before accepting them so you’re aware of any restrictions before betting.

You can now place your initial deposit, which is easy at online betting sites thanks to various deposit and withdrawal options.

To bet on the NBA Rookie of the Year, navigate to the basketball section of the sportsbook website, select the NBA, and go to the futures tab.

You’ll find Rookie of the Year odds, NBA MVP odds, and more there. Select the bet you want to make, enter the amount you want to bet, and finalize your bet
slip.

Ensure you’re using only legal online sportsbooks for any type of betting. Offshore sportsbooks advertise widely on the internet, but they are illegal and offer
no protection against fraud and other criminal activities.

Best sportsbooks for NBA Rookie of the Year odds

Here are some of the best sports betting sites where you can bet on the next NBA Rookie of the Year.
</item 14>

<item 15>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Line shopping is the key that unlocks the best odds and lines at top sports betting sites.

The process is as simple as checking the lines and odds for your prospective bet at multiple online sports betting sites. The result is quite often better odds
and improved lines.

What is line shopping, and why is it worth your time? Find the answers below in our complete guide to line shopping.

How do sportsbooks set odds and lines?

The best sportsbooks employ oddsmakers and oddsmaking services to set their lines and odds.

Other sports betting apps simply copy those lines. The oddsmakers and oddsmaking services use computer models and algorithms that consider analytics,
statistics, injury and roster news, performance, matchup data, and more to develop the lines and odds.

Betting lines and odds are then subject to change based on the betting public’s reaction and any additional information that comes to light ahead of the game.
Sportsbooks will move the line or change the odds if the public bets heavily on one side.
</item 15>

<item 16>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Favorites to win the game are shown negative odds, such as -135, while the underdogs will have positive odds, like +165.

NBA over/under bets: Also known as totals. These bets are focused on the combined point total of the two teams playing in a game. (Winners and losers don’t
matter.) For example, the over/under could be set at 211.5 points in the Knicks vs. Heat matchup. If you were to pick the over on the bet, you’d need the two
teams to combine for 212 points or more to win. They need to score 211 points or fewer if you went with the under.

NBA props and futures

NBA player and team props are growing in popularity. Sportsbooks are taking note and providing more NBA props than ever.

After all, it’s much easier to focus on a single player and how he will perform than the entire team. Player props could focus on how many points a player
scores or how many rebounds he pulls down.
</item 16>

<item 17>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
DraftKings Sportsbook: DraftKings, known as a daily fantasy sports operator before becoming a renowned sportsbook, offers up a nice variety of futures bets,
including the NBA Rookie of the Year award. The online sportsbook and betting app are intuitive and offer an easy-to-understand betting experience.
Additionally, DraftKings fields a competitive welcome bonus for new players.

DraftKings, known as a daily fantasy sports operator before becoming a renowned sportsbook, offers up a nice variety of futures bets, including the NBA Rookie
of the Year award. The online sportsbook and betting app are intuitive and offer an easy-to-understand betting experience. Additionally, DraftKings fields a
competitive welcome bonus for new players. FanDuel Sportsbook: Also known for being in the DFS market, FanDuel has one of the largest followings in the country.
It offers competitive odds for NBA ROTY and has one of the best sports betting apps. Look for a quality welcome bonus for new players and attractive promotions
for existing customers.
</item 17>

<item 18>
Line Shopping in Sports Betting - How to Find the Best Odds (Sporting News, 2023-11-06T22:21:05+00:00)
Sportsbooks want even betting so much that they’ll move a line or change the odds to draw more bets to the other side to get it that way.

That’s when you’ll see a difference between the lines and odds. Every sportsbook jumps on board when a line moves or odds change for a legitimate reason, like
an injury or roster move. It’ll be difficult for you to find value in that case.

But when a line moves or odds change because of betting patterns, only the sportsbook that saw the betting makes the change. That’s your opportunity right
there.

Line shopping is a tool to help you spot changes due to betting patterns. Then, you can simply take advantage of movement in a line or changes in the odds.

Choose a side, shop around at online sportsbooks, compare the lines and odds, and then pounce when you find one that has something different and more favorable
to your viewpoint.

See more: What is a parlay bet?
</item 18>

<item 19>
NBA MVP Odds 2024 Season - Jokic and Giannis early favorites (Sporting News, 2023-10-04T22:11:54+00:00)
Thus, the odds are quite favorable.

While waiting for more information, such as a player’s performance during the regular season, the lines shift and become less favorable. That’s because the
oddsmakers are gathering information and applying new data and analytics to their lines to make them as accurate as possible.

You must decide which approach works best for your betting strategy as a bettor. Do you want to wager when the lines are released with little information at
your disposal but the best potential return on your bet, or do you want to wait until you have more solid information to craft your MVP speculation?

Who decides the NBA MVP winner?

The NBA MVP is selected annually by a panel of sportswriters and broadcasters who vote for the players they believe are the best in the league.

This format, which started in 1980, sees each member of the panel vote for their top three players. Every first-place vote is worth five points, every
second-place is worth three, and third-place is worth a single point.
</item 19>

<item 20>
NBA Prop Bets - Best Team and Player Props 2023/24 Season (Sporting News, 2023-10-02T23:49:35+00:00)
Remember that some of the best players in the NBA were once on the bench for most games until small increases in their playing time gave them a chance to shine.

Shop for the best lines: This tip applies to every bet you make. Using multiple sportsbook accounts lets you find the best odds for every NBA prop bet. Even the
smallest differences, like one sportsbook having odds at +125 and another at +135, can make a difference in your profit. By shopping for the best lines, you can
make smarter bets and find the best value.

Related Pages:
</item 20>

<item 21>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
His wingspan and defensive prowess will challenge opponents. Offensively, he capitalizes on mismatches and has diverse scoring abilities, from the paint to the
3-point line. His underrated passing syncs well with OKC's pace and style.

How to bet on NBA Rookie of the Year

The first thing you need when preparing to bet on the NBA Rookie of the Year Award, or any other bet, is an account at an online sportsbook.

Signing up for a new betting account doesn’t take much time. Once you visit the online sportsbook you’re interested in, click a “sign up” or “register” button.
This will take you to a form for your details. Fill that out and accept the sportsbook’s terms and conditions.

At that point, you’ll be able to claim your welcome bonus, which you’ll qualify for by following the link on this page from our site to the online sportsbook.
If you need to enter a code to get the bonus, you’ll also find that code on our site.
</item 21>

<item 22>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
With legal sports betting now at your fingertips in over 30 states, it’s easier than ever to bet on basketball at one of the best NBA betting sites.

The NBA gives you plenty of games, teams and players to bet on across hundreds of odds markets. As a bettor, you'll want to find the best NBA betting lines
night after night plus NBA futures markets from NBA Finals odds to Rookie of the Year odds.

See below for where to look and more on betting on the NBA at the best sports betting sites.

Best NBA betting sites for 2023

When it comes to betting on the NBA, you can choose any online betting site and have plenty of odds to choose from.

You can bet everything from moneylines, totals, and point spreads to player props for scoring, rebounding, and assists, or futures bets like odd to win the
Eastern Conference or individual award winners like NBA MVP odds.

Here are our favorite sportsbooks for NBA betting.
</item 22>

<item 23>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Season Player Position Team 2023 Paolo Banchero Forward Orlando Magic 2022 Scottie Barnes Forward Toronto Raptors 2021 LaMelo Ball Guard Charlotte Hornets 2020
Ja Morant Guard Memphis Grizzlies 2019 Luka Dončić Guard/Forward Dallas Mavericks 2018 Ben Simmons Forward/Guard Philadelphia 76ers 2017 Malcolm Brogdon Guard
Milwaukee Bucks 2016 Karl-Anthony Towns Center Minnesota Timberwolves 2015 Andrew Wiggins Forward/Guard Minnesota Timberwolves 2014 Michael Carter-Williams
Guard Philadelphia 76ers 2013 Damian Lillard Guard Portland Trail Blazers 2012 Kyrie Irving Guard Cleveland Cavaliers 2011 Blake Griffin Forward Los Angeles
Clippers 2010 Tyreke Evans Guard/Forward Sacramento Kings

What other NBA futures can I bet on?

Rookie of the Year is not the only form of NBA futures betting.
</item 23>

<item 24>
NBA MVP Odds 2024 Season - Jokic and Giannis early favorites (Sporting News, 2023-10-04T22:11:54+00:00)
Popular features include the “edit my bet” and “easy parlay” options, which have helped make BetMGM a leading sportsbook in many states where sports betting is
legal. Caesars Sportsbook: The Caesars betting app may be basic, but the simplicity is part of the draw for customers, along with an outstanding selection of
competitive odds. Caesars also provides NBA betting promos and more sports betting bonuses & promotions for existing customers.

When should you bet on NBA MVP futures?

You can bet on the regular-season NBA MVP award well before the season starts. But when is the best time? That comes down to your tolerance for risk versus
reward.

The odds are so favorable when the lines for NBA Most Valuable Player are first released because the oddsmakers, like you, have very little information to work
with.

They are basing their odds on past performance and expected future accomplishments, as well as the quality of the team around the top candidates for the award.
Thus, the odds are quite favorable.
</item 24>

<item 25>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Rookie of the Year is not the only form of NBA futures betting. Legal online sportsbooks have a wide selection of NBA futures you can bet on, including:

In addition to futures bets, you can wager on many NBA bets during the regular season and postseason. NBA game lines (moneylines, point spreads, and totals) are
the most popular, but you’ll also be able to bet on NBA team and player props, parlays, and more.

The NBA is the second-most popular odds market to bet on in the US, behind NFL betting odds. That means every online betting site always has a great selection
of NBA odds.
</item 25>

<item 26>
Point Spread Betting - What is a Point Spread Bet? (Sporting News, 2023-11-01T22:50:52+00:00)
Do point spread odds change?

Yes, point spread odds can change, and these shifts are commonly referred to as "line movement."

Lines move when the point spread or the odds associated with a particular bet change after sportsbooks have initially set them. There are several reasons for
these adjustments.

One of the most common causes is an imbalance in the amount of money being wagered on one side of the bet. If a significant majority of bettors are placing
their money on one team to cover the spread, the top sports betting apps will adjust the point spread to make the other side more appealing, thereby balancing
the betting action.

Another factor that can affect the spread odds is the occurrence of unexpected events, such as injuries to key players, changes in team strategies, or even
significant weather events that might impact a game's outcome.

For instance, if a star quarterback is announced as injured just days before a big game, the point spread might shift in favor of the opposing team.

Moreover, information and public perception play a role.
</item 26>

<item 27>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Player props could focus on how many points a player scores or how many rebounds he pulls down. These are typically provided in an over/under format.

Then, team props can range from which team will score the first bucket to the total points scored in a quarter to which team will lead at halftime. The
selection is often robust, especially when high-profile teams play.

Futures bets are exactly what their name implies: bets on events that will happen in the future. Common futures markets include:

and much more. You can usually bet on NBA futures odds throughout the year, including well before the season starts and throughout the regular season. The
further away from the event, the more likely you will find heavier odds in your favor.

Is NBA betting legal in the U.S.?

In the US, the landscape of sports betting, specifically NBA betting, has evolved considerably. While NBA betting is permitted, its legality varies across the
nation. The significant turning point came in 2018 when the US Supreme Court made a pivotal decision.
</item 27>

<item 28>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
Either way, the line shopping strategy is designed to help you maximize your potential return.

NBA live betting

Thanks to the increased popularity of online sports betting apps, in-game betting has exploded nationwide. Also known as live betting, in-game betting allows
you to bet on a game while it’s unfolding.

The action is fast, and the odds are constantly shifting, making for a new way to enjoy the flow of a game while still staying engaged as a bettor.

Depending on the sportsbook, there is usually a strong variety of live betting opportunities. You’ll find game and player prop bets like whether the next bucket
will be worth one, two, or three points.

You can bet on the team leading at the end of the current quarter or on whether there will be a comeback victory. You can even boil live betting down to the
results of the next possession.

The most popular sportsbook apps, such as DraftKings, FanDuel, and BetMGM, offer NBA in-game betting, though each has its lines and betting opportunities.
</item 28>

<item 29>
NBA Rookie of the Year Odds - Wembanyama the early favorite (Sporting News, 2023-10-04T22:54:23+00:00)
Look for a quality welcome bonus for new players and attractive promotions for existing customers.

Also known for being in the DFS market, FanDuel has one of the largest followings in the country. It offers competitive odds for NBA ROTY and has one of the
best sports betting apps. Look for a quality welcome bonus for new players and attractive promotions for existing customers. BetMGM Sportsbook: As one of the
most recognizable names in the gambling industry, BetMGM knows how to attract and keep customers with competitive odds for all bet types, including futures bets
and the NBA Rookie of the Year. BetMGM offers many deposit and withdrawal options and 24/7 customer service, and generous sports betting bonuses and promotions.

As one of the most recognizable names in the gambling industry, BetMGM knows how to attract and keep customers with competitive odds for all bet types,
including futures bets and the NBA Rookie of the Year. BetMGM offers many deposit and withdrawal options and 24/7 customer service, and generous sports betting
bonuses and promotions.
</item 29>

<item 30>
The Best NBA Betting Sites and Apps for the 2023-24 Season (Sporting News, 2023-10-02T21:58:37+00:00)
See more: Best NCAA basketball betting sites

2023-24 NBA Season

October 24 Regular season begins November 3 - December 9 In-season tournament February 16-18 All-Star Weekend 2024 April 14 Regular season ends April 16-19
Play-in tournament April 20 Playoffs begin June 6 NBA Finals Game 1

NBA spreads, money lines, and over/unders

For anyone new to NBA betting, the staple wager types are:

NBA point spreads: Bet on games via margins of victory or loss. Say the New York Knicks are -7.5 favorites over the Miami Heat. That means the Knicks need to
win the game by 8 points to win a spread bet. On the other hand, the Heat can lose by up to 7 points (or win outright).

NBA moneylines: The simplest form of betting. You pick the winner of the game, and if you’re right, you’ll get paid out based on the odds when you finalize your
bet slip.
</item 30>
=== END ITEM id=4fd4df3b342b04ef ===

=== ITEM id=1aab8ce95fd9ddaa ===
Gold sentences and facts:
Sentence 0: His career has only blossomed in the years since he took that trip, and while some artists might feel trepidation at the attainment of so much success from such an early age, Mitchell is clear-eyed.
  Fact 0.0: His career has only blossomed in the years since he took that trip.
  Fact 0.1: Some artists might feel trepidation about attaining so much success from such an early age.
  Fact 0.2: Mitchell is clear-eyed.
Sentence 1: Since then he’s been linked to club jobs in Europe but opted to stick around with the national team.
  Fact 1.0: Since then, he has been linked to club jobs in Europe.
  Fact 1.1: He opted to stick around with the national team.

Retrieved items:
<item 1>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
He only fully focused on a potential AFL career last year after previously playing volleyball. Starred with the Calder Cannons and Victoria Metro this year
kicking 33 goals in the talent pathway. His 2km time of 6:05 at the national Draft Combine was a strong indicator of his elite endurance but also his
competitiveness for a player of his size. Is eligible to join the Western Bulldogs under the father-son rule as the son of former Bulldogs utility Matthew
Croft, who played 186 games for the club.

Weapon: Aerial ability

AFL player comparison: Max King (St Kilda)

Jordan Croft has nominated as a father-son prospect for the Western Bulldogs ????
</item 1>

<item 2>
McRae reveals hidden message he carried under collar of his grand final shirt (The Sydney Morning Herald, 2023-09-30T22:45:39+00:00)
“I’m super grateful that I’ve got a little chant like that, and hopefully, it’s a little legacy I can continue, and I’m just happy the Magpie army are looking
after me. I feel like they’re always on my side, no matter what mistake I make ... if anything, they motivate me to continue to take the game on and just be my
true self.” Markov felt like he was cheating on Collingwood when he began training with the Blues, after finding out the Pies overlooked him for what was then
their final list spot.

That followed Gold Coast delisting the 27-year-old, who also played 23 games for Richmond. There was interest in him at the time, but he feared his career might
be over. “I went through the national draft and no one looked into me, so there were technically 17 clubs that didn’t really want me,” he said.
</item 2>

<item 3>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also named captain of the All Australian team after leading the Allies to their first title. Caught the eye with a strong performance for the Australia
under-18 team in its representative match against Port Adelaide’s SANFL side and backed it up in the second match against Carlton’s VFL side where he collected
24 disposals to be one of Australia’s best in victory. Finished the year as a back-to-back Coates Talent League premiership player with the Sandringham Dragons
where he averaged 31 disposals across nine matches.
</item 3>

<item 4>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Uses his dazzling speed to break the lines and kicks the ball with pinpoint accuracy on his preferred left foot. Averaged 32.8 disposals and 5.8 clearances
across four matches for the Allies at the National Championships to earn All Australian honours. Was equally dominant for the Tasmania Devils in the Coates
Talent League having more than 30 disposals in seven of his nine matches and also kicked 12 goals. Showed he could match it against more experienced opponents
by having 20 disposals and three clearances in Australia’s win over Carlton’s VFL team.
</item 4>

<item 5>
How shooting Beyoncé and Kamala Harris made this photographer famous too (The Age, 2023-11-27T23:00:00+00:00)
Tyler Mitchell: “I’ve never been to Australia so it’s very exciting to sort of have a global or new conversation on a new continent with my work.″⁣ Credit:
Courtesy of the artist and Jack Shainman Gallery, New York The theme of the Triennial is “Magic, Matter, Memory”, making Mitchell’s pieces a logical inclusion.
“A lot of my work meditates on being between a heightened reality, or potentially even a magical reality and an actual reality,” he says.

Advertisement

“I’m creating what I call lightly staged moments. And so that means I’m setting up scenarios where these families are enjoying real moments of leisure in
outdoor spaces, usually open pastures and fields, and those moments are meant to stand in for a meditation on what it means to enjoy leisurely outdoor nature as
a black person.” Tyler Mitchell, Vastness, 2021.
</item 5>

<item 6>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Versatility and contested ball-winning ability

AFL player comparison: Callum Mills (Sydney Swans)

"All through my junior football, my dad placed a big emphasis on playing every position. I think that came to the fore this year a little bit."



Tassie’s 'Mr Fix-It' James Leake is shaping as a first-round pick following an impressive season.https://t.co/fwnC96Uzdw — AFL (@AFL) November 3, 2023

Kane McAuliffe

State: South Australia

State League Club: North Adelaide

Community Club: Central Augusta

Date of Birth: 01/03/05

Height: 187cm

A powerful midfielder who wins his own ball and is a penetrating left-foot kick. Is a strong and aggressive tackler as well as being an accomplished extractor
of the ball from stoppages using his clean hands and vision to set up the play.
</item 6>

<item 7>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Horne briefly worked at Saracens as a general manger.

RA always said they wanted to have the role filled by Christmas, with chief-executive Phil Waugh saying the successful candidate had one of the biggest jobs to
change the game’s standing in Australia.

“It’s a big job,” Waugh said. “It’s the biggest job on my leadership team in terms of the impact it has on the general sentiment around the game.”

Horne will oversee all of Australia’s men’s and women’s high-performance programs.

Advertisement

He will also be tasked with “integrating” the programs, with RA trying to centralise all of the Super Rugby’s high-performance programs.

Perhaps most pressing will be building a list of coaching candidates to take over from Eddie Jones, with RA wanting to have the new Wallabies coach set in stone
by the start of Super Rugby.
</item 7>

<item 8>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
His ferocious attack on the ball and his opponent helped set the tone for the Allies to dominate the midfield battle throughout the National Championships and
he was rewarded with All Australian honours. Averaged 29.2 disposals, 11.2 contested possessions and a team-high 6.8 tackles across four matches for the Allies
and showed excellent repeat speed and courage as well as clean hands playing both on the inside and outside. His 6:06 effort in the 2km time trial at the
national Draft Combine showcased his elite endurance. He has been part of the Sydney Swans’ Academy since the under-10s.
</item 8>

<item 9>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
With moments left in extra time against the South Americans, Arnold dragged Mat Ryan and replaced him with Andrew Redmayne – who saved the decisive spot kick
(and also threw his opponents water bottle with cheat sheet attached away from the goal).

Advertisement

Ryan might well have gotten the Socceroos to Qatar, but if Australia had not advanced there’s little chance Arnold would still be their coach today.

And there’s virtually no chance he’ll have to endure a similar level of stress through to the next finals – with FIFA expanding the tournament from 36 to 48
teams meaning eight direct Asian confederation qualifiers will proceed directly instead of four.

Arnold, in his second stint as national team manager, will prowl the technical area for a 59th time on Thursday, taking him clear of Frank Farina.
</item 9>

<item 10>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was also impressive for the Murray Bushrangers in the Coates Talent League, averaging 16 disposals and 3.7 tackles across 10 matches and booted nine goals. His
high natural leap and pace were both prominent at the national Draft Combine with his standing vertical leap of 78cm ranked second overall and his time of 2.96
seconds on the 20m sprint ranked in the top ten.

Weapon: Elusiveness and creativity

AFL player comparison: Tom Papley (Sydney Swans)

Will Graham

State: Queensland

State League Club: Gold Coast Suns Academy

Community Club: Palm Beach Currumbin

Date of Birth: 27/07/05

Height: 186cm

Adapted seamlessly to playing in defence for the Allies after predominantly playing midfield for the Gold Coast Suns Academy.
</item 10>

<item 11>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
NEW PODCAST! Michael Atkinson and Sam Worthington preview a thrilling final round of pool games to decide the quarter-finalists at the Rugby World Cup in
France﻿

He wanted to regain form at club level ﻿with Urayasu D-Rocks but his hand remained up for Test selection.

"I chose to leave Australian shores and understand the ruling around that and what it holds," he said.

"I've got to perform for my club first and if you get the call up, you get the call up. If you don't, I'll take my three months break and enjoy that."

Kerevi is an unabashed Jones fan.

Eddie Jones with fans during a Wallabies training session at Stade Roger Baudras. (Getty)

"He's up at 4am working hard. We catch him in the gym, he's working hard and I swear he doesn't even sleep.
</item 11>

<item 12>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
“Sometimes players need time, they’re human beings, they need to settle, Sonny had a great track record at Hamburg and Leverkusen but there was this narrative
that started to grow internally and externally that he wasn’t a good finisher.

6 Mitchell is becoming an increasingly recognisable face

“I remember telling a coach ‘you’re wrong, his history shows in two different clubs he’s a very astute finisher off both sides’.”

Mitchell remains good friends with Rangnick from their time together at Leipzig, and last season made headlines when he was spotted at a youth team game while
the German was interim coach at Man United.

So astute is Mitchell in his observations, he even laid out a blueprint for United's success while at Monaco.

“I think context is important to measure what Manchester United's current situation is, but I think they do need to get to the point of really putting down what
Manchester United needs to be in the modern game,” he explained.

"We are a long time from the Sir Alex Ferguson era.
</item 12>

<item 13>
UK View: ‘A bridge too far’ – Boycott, former greats tear into ‘dysfunctional’ Poms with World Cup defence all but dead (The Roar | Sports Writers Blog,
2023-10-26T23:01:13+00:00)
It’s such a lame excuse.”

Atherton described the England players as looking ‘dazed and shell-shocked, unable to comprehend what has befallen them’, and wrote in The Times that the Sri
Lankan defeat officially represented the ‘end of an era’ for English white-ball supremacy.

Advertisement

“Each defeat, now four in five matches, has chipped away at the confidence of a group of players at the end of a cycle,” Atherton wrote.

“No batsman looks in any kind of form. Every player has been used; some discarded and used again. They lost faith in their all-rounder strategy before the South
Africa match, making three changes, and then went back to it here, making three more.

“Players sense uncertainty in decision-making – and there has been a lot of uncertainty.

“It was hard to think that they could put in a worse performance than in Mumbai but they managed it with the bat in Bengaluru.”
</item 13>

<item 14>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Continued that form into this season and was outstanding for Western Australia at the National Championships winning his team’s MVP award and All-Australian
honours after averaging 23.5 disposals and 6.5 marks playing as a defender and midfielder. Played a handful of senior games for Claremont and ended the season
being one of his team’s best in a premiership win at under-18 level. His agility, speed for his size and precise kicking are all features of his game.
</item 14>

<item 15>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
But I don't think people really understand demons, everyone's fights in between.

"Eddie's a human as well and at the end of the day it's a results-driven sport, you know, we need results to get that. But he's committed to us and we love that
about him.

"He hasn't thought about going home, he's thought about how we can contribute best and that's what we've done these last three days. I feel for him, you know,
like as a head coach, you maybe can't show that side of himself.

Eddie Jones and Will Skelton talk at Stade Roger Baudras. (Getty)

"But he's been so strong for us and gives us strength to just be positive. And again, we put ourselves in this situation."

Wide World of Sports also asked Kerevi to deliver a message to disillusioned Wallabies fans.﻿

"I think at home there's so many sports going on and some saying rugby's dying, but rugby's massive here overseas," he said.
</item 15>

<item 16>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Advertisement

After a year in where the Wallabies’ detail and nous was clearly absent on the field, the highly detailed Schmidt would help bring rugby smarts to the
Australian program.

Indeed, the manipulation and ability to play both sides of the ruck was clearly visible by Ireland during his Irish tenure.

What’s more, should RA go down a path of bringing Schmidt into the fold, his own relationship with Ireland’s long-time director of high-performance David
Nucifora could see the Australian reconsider his position with the governing body.

Last year Nucifora told The Roar he would unlikely consider returning to RA when he leaves his post with Ireland following next year’s Olympics.

“I don’t think it’d be wise to revisit,” Nucifora said.

“I think that you have a go at something once, you do your best and then if it’s not good enough or it didn’t work you move on, do something else.
</item 16>

<item 17>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
In March last year Graham Arnold seemed a dead man walking – the Socceroos World Cup qualification campaign was lurching towards failure and the manager was
being battered from all sides.

Arnold’s team had finished seven points off automatic qualification for Qatar after a shocking run of one win in seven games left them facing a Asian playoff
against United Arab Emirates and then another against Peru to make the finals.

While rumours swirled and ex-players and fans bayed for his blood, Football Australia took the pragmatic course and stuck by Arnold. It likely had more to do
with the lateness of the hour and financial considerations than any overwhelming faith in Arnie to run around the fortunes.

Facebook Twitter Whatsapp Reddit Email Share

But he did just that with qualification – and an excellent run at the finals – bowing out to the eventual champions Argentina and after nearly taking them to
extra time.
</item 17>

<item 18>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
But after initially moving back into the coaching fray with the Blues in Super Rugby, he joined the All Blacks ahead of last year’s Rugby Championship as New
Zealand Rugby shook the All Blacks coaching team up following their historic series loss to Andy Farrell’s Irish.

Schmidt’s expertise proved telling, with he and Jason Ryan transforming the All Blacks and helping Ian Foster’s side reach the final of this year’s World Cup
after stunning wins over Ireland and Argentina in the knockout stages.

Importantly, the 58-year-old, The Roar understands, is interested in the Wallabies coaching job.

Dan McKellar, Stephen Larkham, Andy Friend and Ian Foster remain interested in the role.

Should RA move on him, Schmidt’s backing would be largely supported by the Super Rugby franchises, with several sources telling The Roar that the experienced
coach is exactly the type of person the Wallabies need.
</item 18>

<item 19>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
He got to answer critics like Mark Bosnich and Craig Foster as well – bluntly rejecting their negativity with the dismissive “who cares?”. That would have
tasted sweet.

On Tuesday, FA CEO James Johnson, who held firm on Arnold 20 months ago, got to congratulate the coach on his impending record-breaking match against Bangladesh
as 2026 World Cup qualifying starts on Thursday.

Advertisement

“Graham is one of the greatest contributors to the Socceroos,” Johnson said in a statement.

“He’s been a constant in the Australian football landscape for more than 30 years as both a player and coach, being a part of some of the Socceroos most
memorable moments.

“Anyone who knows Arnie will tell you he bleeds green and gold. His love for the Socceroos and football in Australia is both infectious and inspiring.
</item 19>

<item 20>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
However, names such as Alli, Heung-Min Son, Toby Alderweireld, Eric Dier and Kieran Trippier set the side up for a bright future, before chairman Daniel Levy
got in Mitchell’s way.

Leaving what he called his ‘dream job’ after less than two years, it was reportedly due to Levy’s over-involvement in negotiations and the recruitment process,
something which wasn’t an issue at his next club – Leipzig.

Ralf Rangnick landed Mitchell and more success followed. Signings such as Dani Olmo, Christopher Nkunku and Amadou Haidara again showcased what Mitchell can do
with the shackles off, and again the results followed.

6 Getting Nkunku from PSG was one of Mitchell's biggest coups Credit: Getty

Many of his signings starred in a run to the Champions League semi-finals in 2020, and in 2022 won the club’s first ever trophy with a DFB Pokal.
</item 20>

<item 21>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
Arnold – who also represented the Socceroos 56 times as a player – has been involved in national team coaching ranks since 2000, first as an assistant to Farina
and then Guus Hiddink, before a brief period as caretaker coach in 2006.

“It’s certainly a proud moment and an achievement that I’ll reflect on when the time is right,” Arnold said.

“I look at some of the names on that list – Farina, Arok, Thompson, Rasic – guys I played under or assisted in the early part of my coaching career, and it
gives you some perspective on what a journey it’s been so far.

Advertisement

“I think back to that caretaker period and how much I learnt about coaching; it was a challenging experience and one that on reflection I probably wasn’t ready
for. I’m grateful for the opportunity I was given and what it was able to teach me ahead of my pathway into club coaching.
</item 21>

<item 22>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
An excellent decision-maker and ball user by hand or foot, he averaged 19.5 disposals, 12.5 contested possessions and 6.5 clearances to finish third in the
Larke Medal voting and took out Victoria Country’s MVP. Further added to his very impressive record by earning selection in the All Australian team for the
second year in a row. Handled the step up to playing senior football with three impressive games in the VFL and was the best player for the Australia under-18
team in its match against Port Adelaide’s SANFL team.
</item 22>

<item 23>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Was impossible to contain during this year’s National Championships booting 14 majors in four matches for Victoria Metro to top the goal kicking. He also
averaged 12.2 disposals and was rewarded with All Australian selection. He further expanded his game by spending more in the midfield and across half back for
the Eastern Ranges and averaged 20 disposals while kicking 11 goals in nine matches in the Coates Talent League.

Weapon: Speed, agility and leap

AFL player comparison: Shai Bolton (Richmond)

Advertisement

This is just what Nick Watson does ????
</item 23>

<item 24>
Eddie Jones says he needs to ‘give myself an uppercut’ but ‘100 percent’ committed to coaching Wallabies (The Roar | Sports Writers Blog,
2023-10-16T22:37:26+00:00)
None of those targets were achieved and Jones changed his tune about building for the future when he named a young team for the doomed Cup campaign – where his
team became the first Australians to bow out of a RWC in the pool stages.

He finished the year with a 2-7 win loss record – the only two victories coming against lowly Georgia and Portugal.

Advertisement

On Tuesday he said he no regrets with leaving experienced players such as Quade Cooper and Michael Hooper out of the side for the tournament.

“Everyone’s got their own judgment and they’re allowed to have that judgment,” he said. “My choice was that I decided to go with youth. And I stand by that
because I think we’ve got a a lot of good young players who are really going to benefit from that experience at the World Cup, and going forward to the 2027
World Cup.”
</item 24>

<item 25>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
The 30-year-old is undoubtedly one of the world's finest centres in peak form but injuries conspired to limit his potential in France.﻿

Samu Kerevi of Australia reacts after the record loss at the Rugby World Cup. (Getty)

"I don't think I performed where I wanted to perform at a World Cup," Kerevi said.

"I hold myself to high standards and I just felt like I could have helped the team more in certain aspects ... I've been out of the game for nine months but I
felt good coming into it ... I just felt like I was blowing a bit, but felt good towards the back end of a couple of games. But again, it's not where I want my
performance to be.

Despite being dropped for Lalakai Foketi for the pool win against Portugal, Kerevi said he had no thoughts of following fellow Fijian-Wallaby ﻿Marika Koroibete
into international retirement.

NEW PODCAST!
</item 25>

<item 26>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
His love for the Socceroos and football in Australia is both infectious and inspiring. That energy and passion is a huge element to the way he’s able to lead
and inspire our nation’s best male footballers in some of our sport’s most challenging arenas.

“On behalf of Football Australia, I’d like to congratulate Arnie on everything he’s achieved in reaching this milestone and we look forward to him continuing to
build on what is already an exceptional legacy.”

In a remarkable few weeks last year Arnold went from on the nose to a man revived. Since then he’s been linked to club jobs in Europe but opted to stick around
with the national team.

While national team coaches make important calls every game, Arnold’s record-breaking achievement is conceivably down to one big ballsy call in the playoff
against Peru.
</item 26>

<item 27>
‘Bleeds green and gold’: Ballsy career-saving call that’s delivered Arnie a record-breaking moment (The Roar | Sports Writers Blog, 2023-11-13T23:01:20+00:00)
“I was able to put lessons from that time into practice, develop and eventually return to the Socceroos much more prepared for the role. It’s been a privilege
to be part of the national team again with the benefit of that experience, and I’m immensely proud of what this group has been able to achieve in the last four
years.

“In my opinion, there’s no greater honour than representing your nation on the international stage. I look at the young players coming through the Socceroos
system now full of pride in the work they’ve put in to get where they are, and the journey that’s ahead of them.

“I’ve always loved being a part of the Socceroos culture and my desire to see Australia fulfil its footballing potential is still driving me to take this group
to further success.”

Australia starts the road to north America in Melbourne at AAMI Park on Thursday night, before travelling to the Middle East to play Palestine next Wednesday
(1.00am AEDT).
</item 27>

<item 28>
The story behind anguished Wallabies star's photo (Wide World Of Sports, 2023-10-07T22:10:40+00:00)
"Last week 45,000 fans (versus Portugal), last night (60,000 in Lyon). Rugby's such a small aspect in terms of Australians but our fans have turned up here,
we've seen them and spoken to them and their unconditional love for us, the fact that they came 20,000km to get here and support us, has been awesome.

"My message is just to understand that we're working extremely hard, there's no stone unturned, especially the youth in this team, to leave the jersey in a
better place and just to stick with us.

"I know the results haven't been coming our way the last couple of years but when it does turn, that's what we're building towards.﻿"
</item 28>

<item 29>
Paul Mitchell signed Tottenham and Chelsea superstars, fell out with Daniel Levy and has tailor-made plan for Manchester United (TalkSport,
2023-10-16T21:00:00+00:00)
Already a top tier European owner of OGC Nice, there has been some success, but nothing compared to their French riviera neighbours, where Ratcliffe appears set
to look for inspiration.

Paul Mitchell recently left the principality of Monaco, but he could well be back there in meetings with Ratcliffe who is planning to appoint a sporting
director.

Mitchell fulfilled that role at Monaco after previous stints at Leipzig, Tottenham, Southampton and MK Dons, becoming one of football’s most respected
executives.

A former player in England’s lower leagues, Mitchell was previously linked with Chelsea and Liverpool, and much like Ratcliffe, he’s also from Greater
Manchester.

Starring mainly for Wigan and MK Dons, it was with the latter where Mitchell was moved upstairs, and made his first big move by spotting future England regular
Dele Alli after being named their head of recruitment in 2010.
</item 29>

<item 30>
RA set to announce local as high-performance director – and how it could help land Schmidt as Wallabies coach (The Roar | Sports Writers Blog,
2023-12-11T22:30:02+00:00)
Advertisement

“What’s changed back there that I could think that, what I can offer, can change what they’re doing still?

“I’ll always love rugby. I’ll always love Australian rugby. Could I see myself working in Australia again? I doubt it.”

But RA have been making moves to try and twist Nucifora’s arm.

While Nucifora was never in the running for the director of high-performance, it’s understood they are trying to get him on board as a consultant when he wraps
up with Ireland.
</item 30>
=== END ITEM id=1aab8ce95fd9ddaa ===
