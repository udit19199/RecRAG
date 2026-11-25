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

=== ITEM id=04090cd24b8c13c8 ===
Gold sentences and facts:
Sentence 0: As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse gases.
  Fact 0.0: The subject referred to as 'they' is never going to replace the need for reducing emissions of CO2 and other greenhouse gases.
Sentence 1: He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about 90%, according to his math.
  Fact 1.0: He stopped flying.
  Fact 1.1: He became a vegetarian.
  Fact 1.2: He ditched gasoline-powered cars.
  Fact 1.3: He drives a Tesla.
  Fact 1.4: He cut his personal emissions by about 90%, according to his math.

Retrieved items:
<item 1>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
So water and oxygen make MXenes less reactive. But scientists think they can overcome this problem by tweaking MXenes’ structure. Capping the materials’ edges
with bulky salts, for instance, could shield MXenes from damage by oxygen.

It’s also hard, right now, to make large amounts of MXenes. In the lab, researchers can make only several kilograms (pounds) in a single batch. Several
companies are working out how to boost batch sizes.

But Thakur points out one irony: Making MXenes for carbon capture can have a big carbon footprint. It takes a lot of electricity to get the heat needed to make
them, she notes. Much of that electricity is made using CO 2 -spewing fossil fuels. It’s important to make sure that the energy used to make these materials
doesn’t spew more CO 2 than the MXenes can later sop up. Otherwise, that would defeat their use in carbon capture.
</item 1>

<item 2>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
But experiments published in BMC Biology with the Mexican cavefish, Astyanax mexicanus, may help reveal the link between diet and the brain.

Cavefish, an unexpected research model

For Masato Yoshizawa, geneticist and neuroscientist at the School of Life Sciences at The University of Hawai‘i Mānoa, the cavefish is a fascinating model
system. While these fish have unique adaptions for the harsh, lightless living conditions of limestone caves in northern Mexico — such as the loss of their eyes
— they also share several basic features with vertebrates, making experimental results with these fish applicable to other species. These features include the
brain region involved in memory and emotion, as well as many genes and the signalling pathways used to control them.

Humans and other vertebrates have more complexity in these systems, and experiments with the cavefish offer a way to strip it away to investigate fundamental
questions of biology. While working with cavefish, Yoshizawa noticed similarities in their behavior and that of patients with autism.
</item 2>

<item 3>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
“Not as a NASA climate scientist. That’s really important for retaining my job.”)

He worries a third arrest could cost him.

“Do I keep doing science? Or do I, you know, keep engaging in risky activism and maybe lose that job?” Kalmus said.

Meanwhile, Kalmus has grown frustrated that more scientists aren’t agitating at protests and willing to face arrest. And his view of our climate predicament has
grown increasingly dark.

“It feels like it’s worse than I thought it would be in 2023,” Kalmus said, pointing to record-low Antarctic sea ice and record-high sea surface and land
temperatures as signs that Earth systems could be shifting more quickly than the scientific community can grasp.

In recent years, climate change has begun to sting Kalmus’ personal life.

In summer 2020, he sickened himself hiking in a California heat wave, then watched a wildfire burn a few miles from his California home, spewing smoke that left
his voice raspy and his head aching.
</item 3>

<item 4>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
“Everything is already spinning out of control,” said Sarah Wilcox-Hughes, dismayed.

Kalmus opened up to the table. He’d been procrastinating for months on finishing the research and he was running into his “psychological limitations.” He felt
he was letting the planet down.

“Anyone want to rub Peter’s shoulders as he writes?” Ethan said.

Later, Kalmus relayed a recent, gloomy conversation with a fellow scientist.

“The Amazon is probably past its tipping point,” Kalmus said. “And I can’t believe that it happened during my lifetime.”

Peter Kalmus sits with his son Zane on a porch swing at the Possibility Alliance. (Evan Bush / NBC News)

Crickets trilled. Roosters crowed. A tear dripped down Ethan’s cheek.
</item 4>

<item 5>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
“No solo necesitamos entender mejor dónde hace calor, sino también cuáles son los factores socioeconómicos para los habitantes de esas zonas”, dijo Whitely
Binder. “Podemos comenzar a analizar lo que hay en las islas de calor y utilizarlo para comenzar a informar las decisiones políticas que vamos a tomar”.

—Rebecca Owen (@beccapox), Escritora de ciencia

This translation by Nel Rodriguez Sepulveda was made possible by a partnership with Planeteando. Esta traducción fue posible gracias a una asociación con
Planeteando.

This news article is included in our ENGAGE resource for educators seeking science news for their classroom lessons. Browse all ENGAGE articles , and share with
your fellow educators how you integrated the article into an activity in the comments section below.

Text © 2023. The authors. CC BY-NC-ND 3.0

Except where otherwise noted, images are subject to copyright.
</item 5>

<item 6>
10 scorching-hot discoveries made about the sun in 2023 (Live Science: The Most Interesting Articles, 2023-12-24T14:00:00+00:00)
However, much more work is needed before any concrete conclusions can be made.

'Ghost particles' could reveal hidden dark matter

Neutrinos spat out by the sun could lead us to a hidden treasure trove of dark matter. (Image credit: Artistic rendering by Christopher Dessert, Nicholas L.
Rodd, Benjamin R. Safdi, Zosia Rostomian (Berkeley Lab), based on data from the Fermi Large Area Telescope.)

The sun may be harboring hidden particles of dark matter in its fiery guts and neutrinos , or "ghost particles" that are spat out by the sun, could be the key
to finding them , a pre-print paper published in August suggested.

Dark matter is an elusive type of matter whose identity is still a mystery to scientists. Dark matter rarely interacts with regular matter or light but when it
does it is theorized to emit neutrinos, which have almost no mass and no electrical charge.
</item 6>

<item 7>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
SP: Some of the chapters focus on dispelling misconceptions people might have about what's down there in the ocean. What, to your mind, is the biggest, most
pervasive myth about the deep sea?

JC: It's the idea that we know almost nothing about it. There's this very popular idea that we know more about the moon or Mars than the deep ocean. That's only
really true for one very specific aspect of knowledge — having a detailed map of the terrain of its solid surface — because the moon and Mars are not covered in
seawater, which blocks radar and means we have to use sonar in the deep ocean. Apart from that, we know far more about the deep ocean than those other places.

SP: The deep sea has attracted a lot of attention recently in the advent of deep sea mining . How worried are you about that?
</item 7>

<item 8>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Kalmus, trailed by a reporter, buzzed around the homestead like a giddy kid, hiking back and forth to a home in an eco-village near the Possibility Alliance
where he was living and working on his research. He tramped down a forest trail in bare feet, paused work for a midday dip in a shady stream and never changed
his T-shirt.

He ground his coffee with a mortar and pestle, thumbed through a book of Enneagram personality types on the porch with his son Zane and listened to Ethan’s
stories about sailing across the Atlantic in a cargo ship to avoid the carbon emissions of flying.

“I feel met by Peter,” Sarah Wilcox-Hughes said of Kalmus. “I don’t feel like there are many people prioritizing the needs of the planet as he is and we are.”

Kalmus found life in Belfast, and at the Possibility Alliance, slower, richer and more connected to the natural world.
</item 8>

<item 9>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Authoritarians know this, and this is why they have troll farms that can manipulate the range of solutions and the sentiments about them.

The Internet as we experience it represents a central obstacle to climate action, through several mechanisms. Promotion of false information about climate
change is only one of them. There’s general political polarization, which inhibits the coalitions we need to build to realize clean energy, as well as creates
paralyzing infighting within the climate movement about strategies, which the platforms benefit from. There’s networked opposition to the infrastructure we need
for the energy transition. There’s the constant distraction from the climate crisis, in the form of the churning scandals of the day, in an attention economy
where all topics compete for mental energy. And there’s the drain of time and attention spent on these platforms rather than in real-world actions.

Any of these areas are worth spending time on, but this essay focuses on how the contemporary media ecology interferes with climate strategy and infrastructure
in particular.
</item 9>

<item 10>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
In other words: less harmful rubber, more grip, better wear, and better range or fuel economy.

The molecular menace

Firestone’s IndyCar tires.

Drive long enough, and the well-defined tread patterns of even the most efficient tires will fade, slowly balding like a stressed-out environmental scientist’s
pate. It’s easy to imagine that the lost rubber simply disappears, but the truth is more terrifying.

Tires break down into tire and road wear particles, or TRWP. This black dust accumulates on roads mostly — but not permanently. When it rains, it is carried
away.

A recent Pew Charitable Trusts report found that 78 percent of ocean microplastics are synthetic tire rubber, which makes efforts to cut down on microfiber
pollution from our laundry seem a bit quaint.
</item 10>

<item 11>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
Several research groups have been exploring MXenes’ potential to capture CO 2 from the air. In the Linköping experiments, “CO 2 stuck to the MXene surface like
refrigerator magnets,” Persson says. “It was absolutely amazing.” His group has shown that MXenes can sponge up as much as half their weight in CO 2 .

One large family

MXenes don’t exist in nature. Scientists have to make them.

This is freshly produced MXene, which normally forms a dark powder when dry. This MXene contains alternating layers of vanadium (M) and carbon (X) atoms.
Karamullah Eisawi/Tulane Univ.

They do this using other labmade compounds called MAX phases. Like MXenes, MAX phases are layered crystals. But MAX phases have extra sheets of metal “A” that
sandwich between their M and X layers.

To turn MAX phases into MXenes, scientists must eat away the A layers using a strong acid.
</item 11>

<item 12>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Feedbacks calculated from observations or from atmosphere-only model simulations forced with the observed surface warming pattern over the past couple of
decades imply less warming than those from model simulations with a fully interactive ocean, which have the freedom to create their own surface warming patterns
[e.g., Gregory et al., 2020].

The third strand of research came from oceanography, showing that the atmospheric cooling effect of ocean heat uptake differs depending on where it occurs: One
unit of ocean heat uptake in high latitudes cools Earth more effectively than the same unit taken up by the low-latitude oceans. This difference is relevant
because the largest heat uptake by the ocean occurs at higher latitudes. The effect, termed ocean heat uptake efficacy, turns out to be another manifestation of
the dependence of radiative feedbacks on surface temperature patterns [Winton et al., 2010; Lin et al., 2021].
</item 12>

<item 13>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins,” says Nick Molden, CEO of Emissions Analytics, an independent group focused on testing and data about the overall environmental impacts of
transportation.

“What you don’t want is all these little wear particles full of all these nasty compounds settling in the environment and inside us and gradually leaching out
their toxins.”

One such toxin is 6PPD-quinone. A 2020 study from the University of Washington’s Center for Urban Waters found that a dramatic decline in coho salmon
populations was the result of the toxin. The salmon have now been placed on some West Coast endangered lists.

“Shortly after birth, they were essentially going mad and dying,” Molden says. “The researchers finally linked it to this compound in tires because it was
happening after rainfall. The rainfall was washing all the accumulated tire material off the road into the harbors and into the shore.”
</item 13>

<item 14>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
And most pressing for the communities relying on the coupled models’ climate change projections, To what extent are the remaining carbon budgets [Zhou et al.,
2021] with respect to global warming targets, rates of near-term future warming, and detection and attribution efforts contingent on the pattern effect?

A Collaborative Outlook

Working to solve the problems outlined above at the intersection of different disciplines and with input from different research communities will likely benefit
related research into, for example, the sensitivity of tropical cyclones and future rainfall changes in the southwestern United States or in South America to
surface warming patterns.

Promising new tools and observations are beginning to emerge. They include targeted model experiments and intercomparisons, extended global satellite
observations of clouds and radiation, Argo floats sampling deep-ocean heat uptake, estimates of radiative imbalances before the year 2000, and new observational
constraints on clouds’ sensitivity to environmental controlling factors. These developments encourage an optimistic outlook on our ability to quantify the
pattern effect and its implications over the next couple of years.
</item 14>

<item 15>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
“For the next five years, if you want to sell tires in California, you need to go through the process of either eliminating 6PPD, or at least properly
researching the alternatives,” Molden says.

“We are collaborating with researchers and other scientists to better understand this transformation product, fill knowledge gaps, and determine next steps,”
says Helsel.

Other initiatives are coming to limit overall tire emissions, including the upcoming Euro 7 regulations, which put caps on tire wear particles as well as other
emissions.

That seems like progress, but Emissions Analytics’ Molden isn’t so sure: “The reason I’m worried about that is I’m sure you can reduce that wear mass, but if
you do that by making the compound more toxic, you could actually make the environmental impact worse.”

“You could actually make the environmental impact worse.”

That could be akin to increasing a car’s fuel efficiency by switching to leaded fuel.

Molden believes that what’s needed is better understanding.
</item 15>

<item 16>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
The scientists found that adoption of the Mediterranean lifestyle was indeed feasible and could be adapted to local contexts, such as availability of
ingredients, their affordability, local culinary traditions, and food preferences. Participants were found to have lowered their risk of cancer and
cardiovascular disease relative to the general population.

“We found that the three blocks of the MEDLIFE were independently associated with lower […] mortality,” explained Sotos-Prieto. “And within block 3 [physical
activity, rest, social habits, and conviviality] we found that participating in collective sports, limiting sedentary activities, and having adequate hours of
sleep (6-8 hours) were associated with lower mortality.”

However, there were some variables that could not be taken into account due to a lack of information. This was the case for olive oil consumption, a prominent
component of the Mediterranean diet previously associated with protective roles in many chronic diseases. “It is probable that we are underestimating the
Mediterranean lifestyle association,” said Sotos-Prieto.

To nap or not to nap?
</item 16>

<item 17>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
Scientists unravel the factors of the Mediterranean lifestyle that lead to reduced mortality and how it can be adopted in non-Mediterranean populations.

Picture this: Olive oil, tomato, garlic, and an abundance of fruits — a delightful communal meal followed by a short nap. Throw in some sun and moderate warm
temperatures and you might find yourself transported to the idyllic Mediterranean.

But what if I told you that you bring the benefits of the Mediterranean lifestyle home with you, even if you live far from the region? Just like a cherished
souvenir brought back from holidays.

“The Mediterranean lifestyle represents a traditional way of living based on a plant-based diet rich in fruits, vegetables, legumes, nuts, whole grains, and
healthy oils (e.g., olive oil), moderated intake of fish, dairy and very low in red and processed meats or sweets along with food habits such as adding spices
instead of salt,” explained Mercedes Sotos-Prieto, assistant professor at the Department of Preventive Medicine, Public Health and Microbiology at the
Universidad Autónoma de Madrid.
</item 17>

<item 18>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
One of the grand challenges in climate science is to reduce uncertainty in estimates of climate sensitivity, which quantifies how much Earth’s surface warms in
response to a doubling of carbon dioxide relative to preindustrial levels. This uncertainty is large because climate sensitivity aggregates myriad processes,
from microscale aerosol-cloud interactions to planetary-scale atmospheric and ocean circulations, into one number. Clouds, which are notoriously difficult to
measure and simulate, are the main driver of the uncertainty.

Various lines of evidence are used to estimate climate sensitivity, including climate model simulations of varying complexity, observations over the past
century, proxies that measure climate change in the distant past, and theory. The likely range of estimates of climate sensitivity was stubbornly constant at a
distressingly imprecise 1.5–4.5 K for several decades, but the research community’s efforts have recently chipped away at this range (Figure 1).
</item 18>

<item 19>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Twice, he’s been arrested during climate protests.

His social media posts often evoke heartbreak.

“We’re in the process of losing basically everything,” he wrote recently on X.

Is anyone listening?

climate scientist natural farming (Evan Bush / NBC News)

Not his suburban neighbors in North Carolina, who continue to drive emissions-belching SUVs to big-box stores. Perhaps not even some in the science community,
like those who kicked him out of the American Geophysical Union’s annual fall meeting last year after he unfurled a banner and protested on the major science
conference’s biggest stage.

He has grown increasingly frustrated with President Joe Biden, who signed the Inflation Reduction Act as his signature climate bill. Kalmus thinks it does too
little to shut out the fossil fuel industry.

(Rhodium Group, a nonpartisan think tank, estimates greenhouse gas emissions will drop 32%-42% below 2005 levels by 2030, well short of Biden’s own benchmark
for progress. The inflation bill is responsible for about a quarter of that projected decrease.)
</item 19>

<item 20>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
They would follow each other in groups and ceased going round in circles. There were some other behaviors, such as attention to a specific task and sleeping,
that were unaffected, but overall the results were promising and according to Yoshizawa, suggest dopamine could be key to how the diet affects behavior.

“We suspect that due to this behavior variation, probably the target of the ketogenic diet is the dopaminergic system,” he said. The challenge now is teasing
out all the complex factors that may be involved.

Diet, environment, and genes – a complex recipe

According to Yoshizawa, there are two plausible ideas as to how the ketones produced by a ketogenic diet are acting on behavior. The first involves the
mitochondria, which use either carbs or fat to produce energy in our cells, and the other involves epigenetics, which simple refers to any non-genetic influence
which turns genes on and off.

Ketones are known to create detectable increases in gene expression in cells.
</item 20>

<item 21>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
One in 10 reported feeling symptoms of anxiety or depression over climate change.

No place is safe

To Kalmus, the best answer he’s found has been to channel his frustrations into protest and civil disobedience, finding community and purpose at the Mountain
Valley Pipeline protests and on the streets of New York City during its Climate Week. Outside of meditation, speaking his mind at these events has been the best
antidote to his climate despair.

When he earned his first arrest, by chaining himself to a bank in Los Angeles last year, he finally felt acknowledged.

climate scientist arrest (Courtesy Brian Emerson/Climate Ad Project)

“It was a good combination of risk and sort of communicating emotional truth. So I was able to dig really deep, and say exactly how I felt in the moment,” he
said.

His outburst went viral. His employer, NASA, sent him a letter expressing concern over his two arrests, he says.

(“I’m speaking on my own behalf and that has to be super clear,” he said in an interview. “Not as a NASA climate scientist.
</item 21>

<item 22>
Creation of quantum dots wins 2023 chemistry Nobel (Science News For Students, 2023-10-04T21:11:57+00:00)
One useful quality that changes with a quantum dot’s size is its color. Usually, “if you want to make different colors with molecules, you would choose a new
molecule,” Linke said. That is, a new set of atoms arranged in a different structure.

But changing a quantum dot’s size can change its color without changing its molecules. Quantum dots give off fluorescent light of different colors when they are
bathed in laser light. Smaller dots give off bluer light. Bigger dots give off redder light.

Dots of the same size made from different materials may give off slightly different colors. Quantum dots are usually made from semiconductors. Such materials
include graphene, selenite or metal sulfides. By adjusting the sizes and materials of quantum dots, chemists can alter their properties for a huge range of
uses.

Possible after all

Scientists suspected that nanoparticles’ sizes could alter their properties nearly a century ago. But at the time, it seemed impossible to actually make such
particles.
</item 22>

<item 23>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
This type of chart illustrates situations in which low values of one variable (in this case, a short nap duration) lead to a decrease in the values of the
second variable (incidence of cardiovascular events), while the trend reverses at higher values.

“Short naps, which range from 0 to 30 minutes, were associated with lower incident cardiovascular disease,” said Sotos-Preito. “An inverse association between
short naps and mortality and cardiovascular diseases was found in our previous publications.”

When asked for a definitive answer as to whether napping was indeed beneficial or not, Sotos-Prieto indicated that more research is needed in this regard.

Adapting to the lifestyle

The challenge remains in how to put these results into practice in other countries with different, deep-rooted customs. Cultural adaptation of beneficial
practices is essential to make the Mediterranean diet and habits more appealing and practical for diverse populations in non-Mediterranean countries.
</item 23>

<item 24>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
The searing temperatures killed a dogwood tree in his front yard. His productivity slowed — he couldn’t focus on science.

Kalmus had dreamed of making a life in the Pacific Northwest, which he felt might be buffered from the worst of climate change. That same year, a heat wave —
considered nearly impossible if not for the influence of climate change — seared the region for three days, killing hundreds of people, buckling roadways and
sending overheated baby birds jumping from their nests and to their deaths.

“That’s when I realized that no place was safe,” Kalmus said. His family relocated to North Carolina, for his wife Sharon’s work, but the experience planted a
seed:

If he couldn’t stop climate change and he couldn’t avoid it, could he at least find a better way to survive it?

Life at the homestead

Kalmus knew what would not work.

“I very quickly eliminated the idea of being a prepper — you know, hoarding beans and ammunition,” he said.
</item 24>

<item 25>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
“Encontramos que la disparidad actual era bastante consistente entre las ciudades: en más del 90% de las ciudades que consideramos, encontramos desigualdades a
la exposición al calor relacionadas tanto en los ingresos como en la raza”, dijo TC Chakraborty, científico de la Tierra en el Laboratorio Nacional del Noroeste
Pacífico y autor principal del estudio. Este tipo de información puede ayudar a los líderes de ciudad a usar mejores medidas de las disparidades del calor, así
como proteger a las poblaciones en mayor riesgo de exposición al calor.
</item 25>

<item 26>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
Otherwise, that would defeat their use in carbon capture.

Educators and Parents, Sign Up for The Cheat Sheet Weekly updates to help you use Science News Explores in the learning environment Client key* E-mail Address*
Go Thank you for signing up! There was a problem signing you up.

MXenes also cost a lot to make. The high cost of making these materials would be less important if they could withstand thousands of CO 2 capture-and-release
cycles. That way, scientists wouldn’t have to keep making new MXenes.

So far, no one has tested MXenes’ durability for stored CO 2 removal and reuse. Still, it helps that MXenes are mechanically rugged. Research has shown that
certain types can be as strong as steel.

The Riverside group’s new paper also points to a related family of nanosheets that may prove good CO 2 sponges. Made from MXenes, they’re known as MBenes,
because the “X” layer is made up of boron atoms.
</item 26>

<item 27>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
Surprisingly effective

Again, Yoshizawa was a bit skeptical. “I doubted it at first,” he said. “I didn’t think it was going to work, but it did.”

In the experiment, cavefish where fed the same ketogenic milk provided to human patients, albeit with a few modifications for fish consumption, and their
behavior was monitored. As a comparison, a type of A. mexicanus fish that lives in rivers and not caves were also tested.

The surface fish do not display the same autism like behaviors as their cave dwelling relatives. In the presence of other surface fish, individuals will begin
to follow each other and swim together, something rarely seen in cavefish, Yoshizawa said. The surface fish also do not do the repetitive behavior of swimming
in circles.

Using these fish as a comparison, Yoshizawa and his students watched and waited. Amazingly, after a month of the ketogenic diet, the cavefish began to act like
the more social surface fish.
</item 27>

<item 28>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He is fixated on them.

Kalmus, 49, thinks civilization is on the path to break down, the Biden administration is clueless on climate, and that he might get fired from his job at NASA
if he is arrested for a third time protesting what he views as downright madness: the continued use of fossil fuels.

Kalmus visited the seaside town of Belfast, Maine, in July and August to visit the Possibility Alliance homestead as he looked for inspiration on how to best
live with the effects climate change. (Evan Bush / NBC News)

“It’s kind of weird and lonely,” he said. “People don’t want to talk about this stuff at parties.”

Kalmus, whose strident tweets reach some 340,000 followers on X, is at the vanguard of a group of scientists who are increasingly disillusioned and growing
convinced that more radical action is needed to wake up political leaders.
</item 28>

<item 29>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Neighbors drop by without notice.

“We’ve created what existed 100 years ago,” Ethan Hughes said.

On a steamy August morning, Kalmus found himself huddled around a faded picnic table at the center of the homestead in Belfast, Maine, sipping his coffee with a
rare breed — people as alarmed by climate change as him.

climate scientist environmental (Evan Bush / NBC News)

Ethan, the charismatic, folksy co-founder of the Possibility Alliance, started the day off with a poem about a deer in the forest and then began to outline
daily chores — chopping wood, building a new deck, caring for goats.

Soon, talk turned to climate change and Kalmus’ grief began to flow as he shared findings from unpublished research on his projections of heat in moderate
emissions scenarios through the year 2300.

“Even in the coolest time of day, people will be dying if they don’t have air conditioning,” Kalmus told the group.

“Everything is already spinning out of control,” said Sarah Wilcox-Hughes, dismayed.
</item 29>

<item 30>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Early in the 2010s, a substantial discrepancy was noted between estimates of climate sensitivity derived from climate models and estimates based on the observed
warming record and radiative balance, the balance between incoming and reflected solar radiation and outgoing terrestrial radiation. Estimates based on observed
warming pointed to much lower values than those derived from models. A key breakthrough toward solving this conundrum has been the recognition of the pattern
effect, the process whereby climate sensitivity depends on the geographic pattern of surface warming. This advance was rated as one of the most promising
avenues for further constraining climate sensitivity in the future [Forster et al., 2021].

Forcing, Feedbacks, and Climate Sweet Spots

Adding greenhouse gases to Earth’s atmosphere leads to a global energy surplus (less terrestrial radiation escapes to space), referred to as forcing. To restore
the energy balance, the planet must warm.
</item 30>

<item 31>
Cavefish provide clues to the keto diet’s effect on autism-like behavior (Advanced Science News, 2023-11-01T08:00:00+00:00)
Ketones are known to create detectable increases in gene expression in cells. Pulling apart exactly how things like, diet, environment, genes and
neurotransmitters are linked is incredibly difficult but could reveal which pathways are best to target for autism treatments or could identify a specific
ketone which works more efficiently than others.

While the cavefish are not an exact model for human autism this work shows how these experiments shine light on the biology and evolution of these behavioral
conditions and offer new directions for researchers to follow.

Reference: Motoko Iwashita, et al. Metabolic shift toward ketosis in asocial cavefish increases social-like affinity, BMC Biology (2023), DOI:
10.1186/s12915-023-01725-9

Feature image credit: Masato Yoshizawa

Correction: This article was edited on November 2, 2023. Ryan Lee was originally described as working at the University of Hawai‘i Mānoa.
</item 31>

<item 32>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
“Zonas con menos árboles en las calles, menos acceso a espacios verdes—Allí es donde se comienza a ver el legado de la desinversión y el racismo institucional
que ha entrado en juego”.

“Zonas con menos árboles en las calles, menos acceso a espacios verdes—Allí es donde se comienza a ver el legado de la desinversión y el racismo institucional
que ha entrado en juego”, dijo Lara Whitely Binder, gerente del programa de preparación climática del condado de King County, Washington, quien no estuvo
involucrada en la investigación.

Debido a que el clima está cambiando y avanzando, de ahora en adelante las ciudades deben prepararse para veranos más calientes y olas de calor mortales más
frecuentes usando distintos medios, tanto en los interiores como al aire libre.
</item 32>

<item 33>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Molden believes that what’s needed is better understanding. To that end, he’s recently created the Tyre Emissions Research Consortium, a LinkedIn group with
about 1,000 members. The intent is to break down the silos that individual researchers were working in.

Research into 6PPD replacements is ongoing, and the US Tire Manufacturers Association is also touting research into various ways to mitigate and reduce tire
debris pollution.

Given all that research is happening at an industry level, it’s easy to feel a little helpless about the role we’re all playing in contributing this pollution
to the environment when we’re just trying to get to work. Still, there are things that we can do to help minimize the environmental impact in the interim.

Molden suggests a few things: drive more slowly, drive more smoothly, and choose the right tires. According to Molden, the tires that your car was outfitted
with when new will provide the best wear characteristics.
</item 33>

<item 34>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
In addition to our growing knowledge of the pattern effect, we also have learned that radiative feedbacks depend on global mean temperature itself: Warming
Earth by 1 K from the LGM emphasizes different feedbacks (e.g., the sea ice albedo feedback) than warming by 1 K from a Miocene hothouse world or warming from 4
to 5 K in a high-emission scenario in a century or two from today (e.g., the water vapor feedback [Bloch-Johnson et al., 2021]). The pattern effect and the
feedback temperature dependence add uncertainties to estimates of climate sensitivity based on the paleorecord, but quantifying their effects would make these
records more relevant to constraining climate sensitivity and expected future warming.

Outstanding questions concern how representative pattern changes in the past century, the past millennium, or quasi-equilibrated times millions of years ago are
of expected future changes.
</item 34>

<item 35>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
But I'm confident that we will see protection for active hydrothermal vents, because we scientists have been saying that for years.

Hydrothermal vents are some of the most vulnerable deep sea habitats and home to 400 animal species not found anywhere else. (Image credit: Universal History
Archive/Universal Images Group via Getty Images)

SP: Deep sea mining is perhaps more manageable in terms of its impacts than other human activities. If not mining, what is the biggest human threat to the deep
sea?

JC: To my mind, it's climate change . And it affects the deep ocean in lots of different ways. The one that concerns me particularly is deoxygenation — the
reduction in oxygen levels — because deep sea animals need oxygen and they get it from the seawater.

Oxygen is carried down by currents that form in the polar regions and sink and spread throughout the deep ocean. As a result of climate change, the ocean is
getting warmer and that means it can't carry as much dissolved oxygen .
</item 35>

<item 36>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
While on his working vacation in Maine, Kalmus invited NBC News for a short visit to the community to see the crisis through the eyes of Americans taking it as
seriously as anyone.

The Possibility Alliance is an 11-acre homestead that operates without power, running water or a sewer system. The Hughes family, which operates the alliance,
pays only one monthly bill for landline phone service. (Evan Bush / NBC News)

Detoxifying

About a decade ago, Kalmus ditched a promising academic career in astrophysics to study the physics of climate change instead.

He made other changes too. He stopped flying, became a vegetarian and ditched gasoline-powered cars (he drives a Tesla), cutting his personal emissions by about
90%, according to his math.

It wasn’t a political statement, but a detoxification.

“Burning fossil fuels started to feel more and more disgusting,” he said.

Ethan Hughes, who operates the Possibility Alliance, rejects globalization and thinks capitalism is a ruinous symptom of “greed and separation.”
</item 36>

<item 37>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Right now, the current plot is about the centralization of discourse on a few corporate platforms. The rise of the platforms brings potential to network
democratic uprisings, as well as buoy authoritarian leaders through post-truth memes and algorithms optimized to dish out anger and hatred. This is a more
challenging story to narrate, because the setting is everywhere. The story unfolds in our bedrooms while we should be sleeping or waking up, filling the most
quotidian moments of waiting in line in the grocery store or while in transit. The characters are us, even more intimately than with climate change. It makes it
hard to see the shape and meaning of this story. And while we are increasingly aware of the influence that shifting our media and social lives onto big tech
platforms has on our democracy, less attention is devoted to the influence this has on our ability to respond to climate change.

Think about these three forces meeting — climate change, authoritarianism, the Internet. What comes to mind?
</item 37>

<item 38>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
DOI: 10.1016/j.mayocp.2023.05.031

Feature image credit: Brooke Lark on Unsplash
</item 38>

<item 39>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Months later, after returning “kicking and screaming” to the “quote-unquote real world,” Kalmus was planning a return trip to the Possibility Alliance over the
winter holidays.

He was still mulling why the Possibility Alliance’s “low energy” lifestyle had felt so meaningful and what was missing from modern life.

Ethan Hughes his goats out to

“Everything is too fast,” Kalmus said. “It’s obvious to a lot of people, but it’s a new revelation to me. There’s nothing more important in life than
relationships.”

This article was originally published on NBCNews.com
</item 39>

<item 40>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
This will include optimizing the production process of hydrogels as well as sourcing affordable raw materials and minimizing waste generated during the
manufacturing of these materials.

“Ensuring the safety and reliability of hydrogel-based batteries is also necessary,” Fan concluded. “Rigorous testing and evaluation of the potential
environmental impact of hydrogel electrolytes, from raw material extraction to disposal, are vital to meeting sustainability goals and gaining public trust.”

Reference: Peihua Yang, Hong Jin Fan, et al., Hetero-Polyionic Hydrogels Enable Dendrites-Free Aqueous Zn-I2 Batteries with Fast Kinetics, Advanced Materials,
(2023). DOI: 10.1002/adma.202306531

Feature image credit: danilo.alvesd on Unsplash
</item 40>

<item 41>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Importantly, the measurement and magnitude of the observed pattern effect are still debated and uncertain. The current best estimate is that the radiative
feedback or damping under idealized, long-term carbon dioxide forcing is 0.5 ± 0.5 watt per square meter per kelvin weaker than the radiative feedback seen
since the late 1800s [Andrews et al., 2022]; 0.5 watt per square meter per kelvin is similar in magnitude to the single radiative feedbacks that the pattern
effect modifies. The wide range of uncertainty opens two contrasting possibilities: The pattern effect might have acted to retard global warming fairly
strongly—in other words, mean global warming might have been much higher by now if the recent surface warming had come about in a different spatial pattern.
This possibility implies that the pattern effect has the potential to influence future near-term warming rates strongly as the surface warming pattern evolves.
</item 41>

<item 42>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
By blending different components, manufacturers can adjust a tire’s grip (how sticky it is), tread life (how long it lasts), and rolling resistance (how
efficient it will make your car). High-grip tires are traditionally short-lived and inefficient, but thanks to sticky science and renewable materials, that’s
changing.

Silica is a good example and an increasingly important component.

Silica is a good example and an increasingly important component. It not only displaces less renewable materials but also makes tires more efficient.
“Conventional tires rely on carbon black to strengthen the rubber matrix as the tire is cured,” Dale Harrigle, Bridgestone’s chief engineer for replacement
tires, says. “In a tire that uses silica in the tread compound, a portion of the carbon black is removed and replaced with silica, which allows for increased
wet performance and reduced rolling resistance.”

Silica traditionally comes from sand, but that’s changing. Goodyear, for example, is now using silica sourced from a rice processing byproduct called husk ash.
</item 42>

<item 43>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The first strand came from analyses of climate feedbacks and sensitivity in model simulations of unequilibrated, transient climate change. If feedbacks were
constant, the expected equilibrium temperature change (climate sensitivity) could be estimated using a very simple energy balance model that linearly
extrapolates the relationship between global temperature change and radiative imbalance [e.g., Gregory et al., 2004]. When longer, fully equilibrated
simulations became available, it became evident that the simple estimation methods assuming constant feedbacks systematically underestimate the actual
equilibrium climate sensitivity. The reason for this underestimation is indeed the evolution of the surface warming pattern, which initially emphasizes more
stabilizing radiative feedbacks but later, during equilibration, emphasizes less stabilizing radiative feedbacks [e.g., Senior and Mitchell, 2000; Rugenstein et
al., 2020].

The second strand related the idea of constant feedbacks to the efforts of estimating equilibrium climate sensitivity from the historical record, as mentioned
above.
</item 43>

<item 44>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
Not visible is the carbon dioxide gas also leaving those stacks. MXene nanomaterials could be installed at or near the top of each stack to collect much of that
climate-warming CO 2 before it escapes into the open air. Miguel Sotomayor/Moment/Getty Images Plus

MXenes are excellent at snagging CO 2 , Persson says. Some researchers think it’s because CO 2 molecules are just the right size to fit between a MXene’s tight
layers. Persson suspects it’s because CO 2 and MXenes like to share electrons, allowing CO 2 to chemically stick to a MXene’s surface.

What’s more, MXenes chemically ignore other common gases — such as nitrogen, the most abundant gas in Earth’s atmosphere. This could allow MXenes to suck
climate-warming CO 2 out of the air while leaving the rest of the atmosphere alone.

Several research groups have been exploring MXenes’ potential to capture CO 2 from the air.
</item 44>

<item 45>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
Roosters crowed. A tear dripped down Ethan’s cheek.

“Let’s dedicate our work today to the Amazon,” Ethan said, outlining for the group how the work planned would sequester carbon and show people they could live
without fossil fuels and that they could take meaningful action against climate change, even as world governments failed them.

(Estimates of when the Amazon could “tip” and transition to savannah vary. Timothy Lenton, a University of Exeter professor who studies climate tipping points
and the Amazon’s resilience, said most estimates suggest the Amazon could cross a threshold as soon as 2 degrees C of warming.)

Life at the Possibility Alliance carried on. Will Foley, who hitchhiked from Florida to stay at the homestead, swung a scythe to clear brush. Frankie Williams
and Max Kurke, recent college graduates living at the alliance as summer apprentices, piled wood while wearing oversize sun hats. Ethan corralled goats for “mob
grazing,” a process he says helps sequester carbon.
</item 45>

<item 46>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Most important, models do not re-create observed cooling in the equatorial and subtropical eastern Pacific and Southern Ocean (Figure 2, right), and it is
concerning that all models have the same sign of error in the trend patterns [e.g., Wills et al., 2022]. In addition, models strongly disagree about the
timescales over which future warming may occur in these regions.

The advent of global coupled storm- and cloud-resolving models, which currently can simulate only a few months to years at a time, presents an exciting
opportunity.

Our limited knowledge of the drivers of surface warming pattern biases in climate models hampers our ability to evaluate these biases. We do not know whether
these models’ inability to reproduce observed trends is due to biases in how they represent the spectrum of internal variability, aerosol forcing,
ocean-atmosphere interactions, deep and shallow atmospheric convection, Southern Ocean cloud-radiative properties, ocean thermocline depth, and ocean
circulation or something else entirely.
</item 46>

<item 47>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
These liquids can be used as fuels to reduce people’s reliance on newly mined fossil fuels, which add new CO 2 to the atmosphere. Or, better yet, these liquids
can be stored somewhere to keep their carbon out of the air for good.

This diagram (cropped for clarity) shows how a dry MXene (upper left) or wet one (lower left) appear to capture and store carbon dioxide molecules (trio of
black and red atoms) within its structure (in right version of each MXene). M. Ozkan et al/Chem. Oct. 4, 2023 ( CC BY 4.0 DEED

Challenges ahead

Since they’re still fairly new, MXenes today are only used in research. Scientists have to iron out some kinks before these will be ready for use in the real
world. And that will take several years, at least.

One weakness of MXenes is that they can break down and transform into other solids called oxides in the presence of oxygen.
</item 47>

<item 48>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
Under a microscope, they look like loose sheets of paper. MXenes become easier to work with when scientists add this powder to water, creating an ink, says
Anupma Thakur. She’s a materials scientist that Purdue University in West Lafayette, Ind.

Mihri Ozkan, an engineer at the University of California, Riverside, is the lead author of the new study showing that certain MXenes can act like carbon-dioxide
sponges. These nano-scale sheets can sop up that climate-warming gas from the air. UC Riverside

As ink, Thakur adds, scientists can simply paint MXenes onto desired surfaces, such as fabric, plastic or glass. Or, if a thick enough coat of MXene paint is
peeled carefully off a surface, it can even become a free-standing sheet.

When toughened with polymers, MXene films can durably separate CO 2 from exhaust gas.
</item 48>

<item 49>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
Climate change isn't going away, and it isn't going to get any better — at least if we keep legislating as we have been. In Democracy in a Hotter Time: Climate
Change and Democratic Transformation, a multidisciplinary collection of subject matter experts discuss the increasingly intertwined fates of American ecology
and democracy, arguing that only by strengthening our existing institutions will we be able to weather the oncoming "long emergency."

In the excerpt below, contributing author and Assistant Professor of Environment and Sustainability at the University at Buffalo, Holly Jean Buck, explores how
accelerating climate change, the modern internet and authoritarianism's recent renaissance are influencing and amplifying one another's negative impacts, to the
detriment of us all.

MIT Press

Excerpted from Democracy in a Hotter Time: Climate Change and Democratic Transformation, edited by David W. Orr. Published by MIT Press. Copyright © 2023. All
rights reserved.
</item 49>

<item 50>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
These materials are harder to make than MXenes. So most research on them has been done with math or computer models. But MBenes could have unique mechanical and
electrical traits, Ozkan’s team finds. And those just might help them last through many, many cycles of CO 2 capture and release.

Each year, human activities spew tens of billions of tons of CO 2 into the air. That’s equivalent to the weight of tens of billions of elephants. Carbon
absorbers, including MXenes or MBenes, can’t catch it all. As such, they’re never going to replace the need for reducing emissions of CO 2 and other greenhouse
gases.

Still, given the urgency of climate change, even a quick or partial fix — such as carbon capture — can’t be ignored. “At this point in time,” says Persson, “we
need to look at all the possible solutions.” And MXene-like materials are emerging as potentially important candidates.
</item 50>

<item 51>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
“Meals are something that happens around family and friends,” she continued. “The conviviality and social aspect are also characteristic. Physical activity is
important in the context of community life, and adequate rest including the typical short mid nap or siesta.”

Sotos-Prieto and researchers from the Universidad Autónoma de Madrid in Spain and the Harvard Chan School of Public Health in the USA are exploring the
feasibility and impact of adopting the Mediterranean diet and lifestyle among non-Mediterranean populations using locally available products and within their
own cultural contexts.

“We are seeing the transferability of the lifestyle [outside the Mediterranean] and its positive effects on health,” she added. The most significant perk? A 29%
lower risk of mortality in both men and women.

Breaking down the effects

Scientists stress that the Mediterranean diet should be understood more as a way of life than a mere dietary pattern with scientific evidence acquired over the
last four decades establishing its benefits on life expectancy, quality of life, and the prevention of several chronic diseases.
</item 51>

<item 52>
'They seemed primed to take over': How the Great Dying doomed the 'beast tooth' and set the stage for the dawn of the dinosaurs (Live Science: The Most
Interesting Articles, 2023-10-01T08:00:25+00:00)
Inostrancevia is a genus from the extinct group Theriodontia that appeared during the Middle Permian. (Image credit: Stocktrek Images/Getty Images)

The excerpt below is taken from "Our Fragile Moment: How Lessons from Earth's Past Can Help Us Survive the Climate Crisis" (Hachette Book Group, 2023), by
Michael Mann. It looks at how climate change following the Cambrian explosion caused the biggest mass extinction on Earth — dooming the creatures set to
dominate and set the stage for dinosaurs to rule.

The mechanisms that can freeze the planet, as was the case with Snowball Earth can also lead to inhospitably hot climates, when enough carbon dioxide enters the
atmosphere. Arguably the greatest extinction event of all time — called the Great Dying — appears to have resulted, at least in part, from a massive
heat-inducing release of carbon into the atmosphere 250 million years ago.
</item 52>

<item 53>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
For example, the cooling pattern during the Last Glacial Maximum (LGM) differed from the warming pattern observed currently and from those expected for the next
couple of decades and expected in the equilibrium following a doubling of carbon dioxide, not only in sign but also in the spatial distribution of magnitudes.
The LGM currently provides the best constraint on the upper bound of equilibrium climate sensitivity [Sherwood et al., 2020], and hence, the details of the
pattern effect matter greatly.

Outstanding questions concern how representative pattern changes in the past century, the past millennium, or quasi-equilibrated times millions of years ago are
of expected future changes.
</item 53>

<item 54>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
What comes to mind? If you recombine the familiar characters from these stories, perhaps it looks like climate activists using the capabilities of the Internet
to further both networked protest and energy democracy. In particular, advocacy for a version of “energy democracy” that looks like wind, water, and solar;
decentralized systems; and local community control of energy.

In this essay, I would like to suggest that this is not actually where the three forces of rising authoritarianism x climate change x tech platforms domination
leads. Rather, the political economy of online media has boxed us into a social landscape wherein both the political consensus and the infrastructure we need
for the energy transition is impossible to build. The current configuration of the Internet is a key obstacle to climate action.

The possibilities of climate action exist within a media ecosystem that has monetized our attention and that profits from our hate and division. Algorithms that
reap advertising profits from maximizing time-on-site have figured out that what keeps us clicking is anger.
</item 54>

<item 55>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Acknowledgments

The authors thank all participants of the US CLIVAR pattern effect workshop. M.Z.’s work was supported by the U.S. Department of Energy (DOE) Regional and
Global Model Analysis program area and was performed under the auspices of DOE by Lawrence Livermore National Laboratory under contract DE-AC52-07NA27344. P.C.
was supported by the UK Natural Environment Research Council, grants NE/T006250/1 and NE/V012045/1. T.A. was supported by the Met Office Hadley Centre Climate
Programme funded by the Department of Business, Energy and Industrial Strategy and received funding from the European Union’s Horizon 2020 research and
innovation program under grant agreement 820829. M.R. was supported by NASA under grant 80NSSC21K1042.

References

Andrews, T., et al. (2022), On the effect of historical SST patterns on radiative feedback, J. Geophys. Res.
</item 55>

<item 56>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The three strands of research have converged over the past few years, highlighting that understanding the pattern effect benefits from—and perhaps
requires—contributions from virtually all climate research communities studying large-scale ocean-atmosphere coupling and the dynamics that set regional to
global responses to external forcing.

To foster this work across communities, 140 scientists from around the world and from different disciplines gathered for a US CLIVAR (U.S. Climate Variability
and Predictability Program) workshop in May 2022 in Boulder, Colo. Below, we present the current consensus that emerged during the workshop and raise questions
that require urgent attention from scientists.

The Past Is a Poor Analogue for the Future

Research on the pattern effect has exposed limitations of using equilibrium climate sensitivity to constrain future warming and, conversely, of using recently
observed warming to constrain equilibrium climate sensitivity. The foremost implication of the pattern effect is that historically observed climate change does
not constrain the upper limit of climate sensitivity [Sherwood et al., 2020; Forster et al., 2021].
</item 56>

<item 57>
La desigualdad del estrés por calor (Eos: Earth And Space Science News, 2023-09-26T12:15:35+00:00)
“Este resultado fue muy interesante – que este nivel de desigualdad y segregación se correlacionara tan fuertemente con los niveles de disparidad de calor,”
dijo.

Proteger a los más vulnerables

“Estudios como este, que buscan comprender más a fondo las disparidades existentes en la exposición al calor, son importantes para identificar qué comunidades
corren mayor riesgo de sufrir cambios ambientales relacionados al cambio climático, como el calor extremo que cada vez empeora más”, dijo Tummala. Se necesitan
medidas de salud pública y estrategias de mitigación para proteger a los ciudadanos con mayor riesgo de sufrir calor – especialmente cuando el aumento de las
temperaturas se vuelve más común.
</item 57>

<item 58>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He’s trying to build a gift economy in Belfast, in which neighbors share resources and services with no expectation of repayment. “I’m trying to role-model
post-petrol,” Hughes said. (Evan Bush / NBC News)

Kalmus, now a married father of two teenagers, is introspective, earnest and narrowly focused on climate change in both his personal and professional life. Tall
with an athletic build, a neat beard and a tangle of messy hair that has a skunklike streak of white, he can sometimes come off like an overgrown teenager: He’s
the smartest person in the room, brimming with idealism and still indignant — with a zeal many adults can no longer muster — about how life’s most annoying
systems work (like the U.S. medical system, a symptom of what he calls the “capitalist death cult”).

The changes he made weren’t enough. Kalmus wrote a book on climate change and chained himself to a Chase bank in protest of fossil fuel financing. Twice, he’s
been arrested during climate protests.
</item 58>

<item 59>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Strong warming in regions of deep convection such as the western Pacific is communicated throughout the troposphere and leads to strong warming aloft, enhancing
radiative emission to space. In contrast, parts of the eastern Pacific have cooled. Warming aloft and cooling at the surface enhance the lower tropospheric
inversion strengths in the eastern Pacific and promote extensive shallow cloud cover that reflects solar radiation and keeps Earth relatively cool. At right is
a typical representation of climate change simulated with coupled climate models, which create their own surface warming pattern that differs from the observed
one. In coupled model simulations, warming aloft is less pronounced, and relative to the observed case, the lower tropospheric inversion strength is decreased,
and cloud cover is reduced through time. Both factors lead to less efficient cooling near Earth’s surface in models than in nature.

Feedbacks involving clouds and the atmospheric temperature structure are most sensitive to spatial differences in warming.
</item 59>

<item 60>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
MXenes could be heated up to release their CO 2 for long-term storage somewhere else. Flushing MXenes with a different kind of gas, such as hydrogen, might also
remove the CO 2 from them. Then the MXenes could be reused for a new round of carbon capture.

MXenes could also help transform the CO 2 they capture into other useful compounds, Stavila says. That’s because MXenes are catalytic. That means they can help
drive other chemicals, such as CO 2 , to react. MXenes also conduct electricity. Since they like sharing electrons with CO 2 , they can ferry those electrons to
other compounds.

As a result, “you can do [transformations of CO 2 ] that may not be otherwise possible,” Stavila says. For instance, with a boost from MXenes, CO 2 can react
with hydrogen to form methanol or formic acid.
</item 60>

<item 61>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
Each MXene has slightly different properties, from how well it conducts electricity to how strong it is.

Scientists can even modify existing MXenes by treating them with different chemicals. This involves covering the MXene’s surface with new atoms, such as oxygen
and fluorine, to change its behavior. For more than a decade, scientists have been tailoring different MXenes for use in batteries and other energy storage
devices. But tweaking MXenes’ chemistry can also alter their ability to collect CO 2 .

In an October 4 paper in Chem, engineers at the University of California, Riverside, reviewed the potential widespread use of MXenes. In a statement, lead
author Mihri Ozkan noted, “Their unique properties make them excellent candidates for capturing CO 2 .”

Painting on a CO 2 -absorber

Freshly prepared MXenes appear as a dark-colored powder. Each fleck contains many MXene sheets. Under a microscope, they look like loose sheets of paper.
</item 61>

<item 62>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
During his time in Belfast, Kalmus strummed a mandolin in a Celtic folk music group that meets at the eco-village, along with his son Zane, Wilcox-Hughes and
her daughter, Etta. He picked berries and ate fresh goat cheese and helped Ethan load hay into a neighbor’s barn.

climate scientist music group (Evan Bush / NBC News)

He spoke at the local library about climate change. He shared art, food and labor with neighbors. He saw how a community was sprouting up around the Possibility
Alliance — people willing to sacrifice the conveniences of a modern life built on fossil fuels.

He felt safer here and more energized to work.

It wasn’t about the place, but the people, he realized.

People at the center

Maine, like anywhere else, isn’t immune to climate change. But here, at least, he had people who understood his angst, who could share music and joy, who were
willing to make their own sacrifices, and who were trying to prepare their community in the way they thought was best.
</item 62>

<item 63>
Strategizing for sustainable development (Business World, 2023-11-05T11:26:10+00:00)
The government calls for embedding resilience, sustainability, and nature-based solutions in infrastructure planning and investment to enhance climate
resilience. Likewise, investments in renewable energy are expected not only to enhance energy security but also reduce greenhouse gas (GHG) emissions.

From an economic vantage point, GHG emissions are economic “bads” and are a cause of inefficiency as they get over-produced. Their effects, however, go beyond
national borders, and while the Philippines contributes minimally to global GHG emissions, the World Risk Index 2022 report identifies the Philippines as the
most disaster-prone country in the world.

The government envisions that by 2028: (1) Climate and disaster risk resilience of communities and institutions will increase (2) Ecosystem resilience will be
enhanced, and (3) A low-carbon economy transition is enabled.

To ensure that these are realized, the government is set to safeguard cross-sectoral convergence and implement a comprehensive risk management approach to
address the adverse consequences posed by climate change.
</item 63>

<item 64>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Res. Atmos., 127, e2022JD036675, https://doi.org/10.1029/2022JD036675.

Bloch-Johnson, J., et al. (2021), Climate sensitivity increases under higher CO 2 levels due to feedback temperature dependence, Geophys. Res. Lett., 48,
e2020GL089074, https://doi.org/10.1029/2020GL089074.

Forster, P., et al. (2021), The Earth’s energy budget, climate feedbacks, and climate sensitivity, in Climate Change 2021: The Physical Science Basis. Working
Group I Contribution to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change, edited by V. Masson-Delmotte et al., pp. 923–1,054,
Cambridge Univ. Press, New York, https://doi.org/10.1017/9781009157896.009.
</item 64>

<item 65>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Author Information

Maria Rugenstein (maria.rugenstein@colostate.edu), Colorado State University, Fort Collins; Mark Zelinka, Lawrence Livermore National Laboratory, Livermore,
Calif.; Kristopher B. Karnauskas, University of Colorado Boulder; Paulo Ceppi, Imperial College London, London; and Timothy Andrews, Met Office Hadley Centre,
Exeter, U.K.

Citation: Rugenstein, M., M. Zelinka, K. B. Karnauskas, P. Ceppi, and T. Andrews (2023), Patterns of surface warming matter for climate sensitivity, Eos, 104,
https://doi.org/10.1029/2023EO230411. Published on 31 October 2023.

Text © 2023. The authors. CC BY-NC-ND 3.0

Except where otherwise noted, images are subject to copyright. Any reuse without express permission from the copyright owner is prohibited.
</item 65>

<item 66>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
The shorter format is an opportunity to focus and update the information — there have been quite a few discoveries in lots of different aspects of deep sea
biology since I wrote "Ask an Ocean Explorer."

We know far more about the deep ocean and its inhabitants than about the moon or Mars, says Jon Copley. (Image credit: Giordano Cipriani via Getty Images)

SP: Research has made great strides in recent years, I'm sure. What are some of the most exciting, new discoveries you discuss in the book?

JC: We're finding out a lot more about how deep sea animals interact with each other and their environment. An area where we've seen a lot of interesting papers
over the past five years has been in sensory ecology — realizing how animals perceive their environment, how they respond to that, how they avoid being seen by
predators... It was nice to bring some of those together in a couple of the chapters.
</item 66>

<item 67>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
To restore the energy balance, the planet must warm. But warming causes changes in the climate system: The concentration of water vapor—a greenhouse gas—in the
atmosphere increases, the spatial coverage of highly reflective snow and sea ice decreases, and cloud properties change. These and other radiative feedbacks
amplify or dampen how much the planet warms in response to the forcing. Hence, for a given forcing, the feedbacks determine the climate sensitivity.

A degree of global warming spread out evenly will cause a different radiative response than if that same warming is concentrated in a climate sweet spot.

For decades, researchers assumed that global mean radiative feedbacks mostly depend on global mean temperature [Gregory et al., 2004].
</item 67>

<item 68>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The reason for the poor predictive power of the climate record from past decades is that the observed surface warming pattern caused feedbacks that were more
stabilizing than the ones projected for the future. The warming was particularly pronounced in the sweet spots of the western Pacific and the subtropical
eastern Pacific—a perfect combination for enhanced radiative damping (Figure 2, left). In particular, low-lying cumulus clouds in the eastern Pacific covered a
larger area and reflected more sunlight back to space than studies suggest they will in the future.

Difficulty in quantifying the pattern effect stems from uncertainties in observed surface temperature trends, the magnitude of ocean heat uptake, and the short
observational record of Earth’s energy budget.

Importantly, the measurement and magnitude of the observed pattern effect are still debated and uncertain.
</item 68>

<item 69>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
However, they also depend on the spatial pattern of surface warming: Much like applying a force uniformly over someone’s entire body will elicit a very
different reaction than tickling the soles of that person’s feet, a degree of global warming spread out evenly will cause a different radiative response than if
that same warming were concentrated in a climate sweet spot (a location where surface warming produces efficient radiative damping).

A wide variety of processes affect the evolution of surface temperature, from greenhouse gas forcing and regional aerosol forcing to natural oscillations
involving the ocean and atmosphere to the continental boundary conditions and the extent of ice sheets and sea ice. The pattern of surface temperature change
over the past 40 or so years featured a pronounced spatial structure, with some locations even cooling in spite of the global mean warming on the order of 1 K
(Figure 2, bottom left).

Fig. 2. At left is the surface and vertical structure of warming observed over the past few decades in the tropical Pacific.
</item 69>

<item 70>
10 scorching-hot discoveries made about the sun in 2023 (Live Science: The Most Interesting Articles, 2023-12-24T14:00:00+00:00)
Researchers hypothesized that the sun's core may have a high concentration of dark matter and, if this is the case, they predict that it will occasionally emit
slightly more neutrinos than normal. However, these extra neutrinos have not been detected so far.

Ancient anomaly in sun's solar cycle

Ancient texts from Korea suggest the sun's solar cycle used to be much shorter. (Image credit: Yan et al. 2023)

It currently takes around 11 years for the sun to complete one solar cycle, during which it transitions from solar minimum to solar maximum and back again. But
a pre-print paper published in June suggests that past cycles may have been much shorter.

Researchers studied ancient texts unearthed in Korea, which contain detailed accounts of auroral displays during the Maunder Minimum — a period of reduced solar
activity between 1645 and 1715. They found that solar cycles during this time likely lasted only eight years .

However, scientists have not been able to properly explain why the solar cycle shortened during this period.
</item 70>

<item 71>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
In a new paper published in the journal Advanced Materials, Nanyang Technological University (NTU), School of Physical and Mathematical Sciences professor
Hongjin Fan and his co-authors examine different electrolytes for aqueous zinc ion batteries that take the form of hydrogels — three-dimensional network
structures that are capable of storing large amounts of water — to address some of these challenges.

“The core concept of our research is that we design the hydrogel electrolytes by introducing functional groups [specific groupings of atoms within molecules
that have their own characteristic properties] that carry either positive or negative charges,” Fan said. “Our idea is to use a hetero-hydrogel to mitigate
challenges on both the anode and the cathode sides of the zinc iodine batteries.”

A hydrogel boost

Fan explained that the purpose of this hydrogel modification is to ensure that when zinc accumulates on the anode side, it does so evenly.
</item 71>

<item 72>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Winton, M., K. Takahashi, and I. M. Held (2010), Importance of ocean heat uptake efficacy to transient climate change, J. Clim., 23, 2,333–2,344,
https://doi.org/10.1175/2009JCLI3139.1.

Wills, R. C. J., et al. (2022), Systematic climate model biases in the large-scale patterns of recent sea-surface temperature and sea-level pressure change,
Geophys. Res. Lett., 49, e2022GL100011, https://doi.org/10.1029/2022GL100011.

Zhou, C., et al. (2021), Greater committed warming after accounting for the pattern effect, Nat. Clim. Change, 11, 132–136,
https://doi.org/10.1038/s41558-020-00955-x.
</item 72>

<item 73>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
The idea that chemosynthesis — where life is powered by a form of chemical energy instead of sunlight with photosynthesis — could sustain whole populations of
animal species was impossible, until we discovered hydrothermal vents and other, similar habitats.

Deep sea vents also glow very faintly — too faintly for the human eye to see, but bright enough that microbes can use it as an energy source. Again, it expands
our notion of what's possible in the cosmos, because you don't necessarily have to be that close to a bright star, potentially, to sustain life.

"Deep Sea: 10 Things You Should Know" is available in the U.K. to order on Amazon .

This interview has been condensed and lightly edited for length.

Deep sea creatures have evolved extreme strategies to cope with their environment. Read about the trials and tribulations of their sex lives in this excerpt
from "Deep Sea: 10 Things You Should Know."
</item 73>

<item 74>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Part of / We only get one planet

Tires are the most important safety component on a car, yet there’s increasing research that they’re doing a good deal of harm, too. Slowly, surely, one mote of
black dust at a time, tires are leaving their mark on the environment.

According to researchers at Imperial College London, our cars and trucks emit 6 million tons of tire wear particles every year, transportation detritus that
goes into the air, the land, and the water. That’s a massive figure that ignores the nearly 250 million tires that are disposed of annually in North America,
roughly half of which are burned.

It’s a depressing state of affairs, but there is room for optimism. Every major tire manufacturer globally is raising its game — not only in terms of
performance in the face of bigger, heavier electric cars but also in terms of environmental impact. Renewable materials are finding their way into modern tires,
and there’s even greener rubber on the horizon, too.
</item 74>

<item 75>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
Chief among them? Location: “Hevea trees are primarily grown in Southeast Asia,” he says, “while guayule is grown in the Southwestern US.”

This is also a climate resiliency play. Guayule provides an alternate drought-resistant source for natural rubber at a time when global temperature patterns are
behaving like the stock market: highly unpredictable but trending upward.

Virtually every tire manufacturer is making at least token efforts to increase the usage of sustainable materials like guayule. Yokohama is using motorsports to
develop its Advan A005 tire, which debuted at last year’s Pikes Peak International Hill Climb and contains 33 percent sustainable materials, including
biomass-derived rubber. Michelin, meanwhile, shod Porsche’s prototype electric GT4 e-Performance racer with tires said to be 53 percent renewable, like strap
steel and lemon peels.

That all may sound like marketing plays — and it partly is — but motorsports have historically proven an ideal development opportunity in the automotive
industry.
</item 75>

<item 76>
Tires are saving us — and killing us, too (The Verge, 2023-11-17T13:45:00+00:00)
The bad news is that we’re still going through more tires than ever. Harrigle says we scrap 250 million tires every year in North America. Again, about half are
burned for fuel.

But there are other solutions. Recycling is one, but reuse is better, a process called retreading. Retreading itself is nothing new, but while it’s generally
fallen out of favor for consumer tires, on the industrial side of things, it can be a huge cost-saver — and environment-saver, too.

Jason Roanhouse, VP of North American Bandag operations at Bridgestone Mobility Solutions, says that the company’s retread business has saved 300 million tires
from being thrown out, worth roughly 4 billion gallons of oil.

Future tires

While efforts to find sustainable replacements for tire manufacturing compounds continue, as does research into what today’s compounds do to our health, many
manufacturers are trying to change the equation altogether.

A near-future example is the Michelin Uptis, a non-pneumatic tire.
</item 76>

<item 77>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
For comparison, a sheet of paper is 100,000 times thicker.

MXenes normally come as stacks of these individual sheets. The many separate sheets inside a stack add up to a huge overall surface area. It’s like how the
combined surface area of all the pages in a book is much larger than that of just the covers.

In fact, the total surface area contained in one gram (1/400 ounce) of MXene “can cover a football field,” notes Per Persson. He’s a materials scientist at
Linköping University in Sweden.

Atoms across MXenes’ large surface areas can interact with their environment. That includes grabbing carbon dioxide, or CO 2, out of the air around them — a
process called carbon capture.

The white exhaust leaving smokestacks at this coal-fired power plant in Arizona is largely steam. Not visible is the carbon dioxide gas also leaving those
stacks.
</item 77>

<item 78>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
But he found himself daydreaming of a simpler life, where he could keep bees, grow vegetables and press cider to drink on a Friday night and live closer to the
land.

Visiting the Possibility Alliance, a sprawling 11-acre homestead filled with fruit trees, goats, chickens and a garden, allowed him to scratch an itch he’d
entertained for much of his adult life.

The Hughes family, which operates the homestead, and its guests live largely without electricity or modern technology.

The family of four doesn’t fly because of climate concerns and also doesn't own a car. Their main use of fossil fuels is to take passenger trains to climate
protests.

They try to eschew capitalism and have built, instead, a "gift economy" where resources are shared and skills are traded among neighbors in this tiny corner of
Maine. They grow much of their own food, hold trainings for climate protesters and plan to host refugees as climate disasters worsen.

Nights are illuminated by candlelight. Neighbors drop by without notice.

“We’ve created what existed 100 years ago,” Ethan Hughes said.
</item 78>

<item 79>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The other end of the uncertainty range implies that the pattern effect might have been negligible over the past century and that it might apply only over
shorter timescales or in drastically different climate states.

Difficulty in quantifying the pattern effect stems from uncertainties in observed surface temperature trends, the magnitude of ocean heat uptake, and the short
observational record of Earth’s energy budget. Our incomplete quantitative understanding of how clouds react to their environment and how these dependences are
represented in the highly parameterized climate models further limits our ability to quantify the full range of the possible and recently realized pattern
effect.

Pressing, fundamental questions for climate dynamics concern how surface temperature patterns come about, how Earth’s radiation budget depends on the details of
the surface temperature patterns, and the extent to which the two depend on each other.
</item 79>

<item 80>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
When water is warmer, the metabolism of things living in the water runs faster and they use up oxygen more quickly, so that makes the problem even worse. And
thirdly, we know that the currents carrying oxygen down to the deep ocean are weakening , because melting ice sheets are making the water fresher and blocking
the formation of dense water than sinks.

Those currents take centuries to complete their journey, which means the changes we have already made are going to carry on being felt for centuries. The deep
ocean is already on track to have 10% less oxygen overall globally than it did in preindustrial times by 2400. It's hard to predict what the knock-on effects
are going to be, but they are going to be widespread and they are coming.

SP: You dedicate much of your time to communicating deep sea science with lay audiences. Why is that so important to you?

JC: I enjoy talking to people about the deep sea because it's not somewhere we think about every day.
</item 80>

<item 81>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Rugenstein, M., et al. (2020), Equilibrium climate sensitivity estimated by equilibrating climate models, Geophys. Res. Lett., 47, e2019GL083898,
https://doi.org/10.1029/2019GL083898.

Senior, C. A., and J. F. B. Mitchell (2000), The time-dependence of climate sensitivity, Geophys. Res. Lett., 27, 2,685–2,688,
https://doi.org/10.1029/2000GL011373.

Sherwood, S. C., et al. (2020), An assessment of Earth’s climate sensitivity using multiple lines of evidence, Rev. Geophys., 58, e2019RG000678,
https://doi.org/10.1029/2019RG000678.
</item 81>

<item 82>
10 scorching-hot discoveries made about the sun in 2023 (Live Science: The Most Interesting Articles, 2023-12-24T14:00:00+00:00)
"A similar solar storm today would be catastrophic for our modern technological society," the researchers wrote.

Ancient 'superflares' may have sparked life on Earth

Ancient "superflares" could have sparked life on Earth, researchers claim. (Image credit: NASA's Goddard Space Flight Center)

Continuing the ancient solar storm trend, a May study revealed that "superflares" from the early hyperactive sun could have provided the energy needed to spark
life on Earth billions of years ago.

Researchers recreated Earth's early atmosphere in the lab and fired charged particles, like those found in solar wind, at the primitive gasses and found that
this created both amino acids and carboxylic acids — building blocks for proteins and all organic life.

Similar experiments recreating lightning have also created these compounds in the lab. But the team argues that lightning strikes would not have provided
anywhere near as much power as superflares, which makes them a less suitable candidate for kickstarting life on Earth.
</item 82>

<item 83>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Gregory, J. M., et al. (2004), A new method for diagnosing radiative forcing and climate sensitivity, Geophys. Res. Lett., 31, L03205,
https://doi.org/10.1029/2003GL018747.

Gregory, J. M., et al. (2020), How accurately can the climate sensitivity to CO 2 be estimated from historical climate change?, Clim. Dyn., 54, 129–157,
https://doi.org/10.1007/s00382-019-04991-y.

Lin, Y., et al. (2021), The dominant contribution of Southern Ocean heat uptake to time-evolving radiative feedback in CESM, Geophys. Res. Lett., 48,
e2021GL093302, https://doi.org/10.1029/2021GL093302.

Rugenstein, M., et al.
</item 83>

<item 84>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
Feedbacks involving clouds and the atmospheric temperature structure are most sensitive to spatial differences in warming. Deep convection in the warmest
tropical regions readily communicates surface conditions upward throughout the troposphere (up to about 10–15 kilometers) and then horizontally across much of
the globe, making the western Pacific a climate sweet spot. This warmer air sitting atop the relatively cool waters in the eastern Pacific or Southern Ocean
acts to stabilize the lowermost troposphere, allowing more extensive low-lying stratus and stratocumulus clouds to develop. Because of their location and
structure, these low clouds efficiently cool the planet and offset some of the initial warming (Figure 2, top left).

Approaching the Problem from Different Angles

Three strands of research have converged over the past few years, highlighting that understanding the pattern effect benefits from contributions from virtually
all climate research communities.

Historically, three strands of research have highlighted the dependence of radiative feedbacks on the surface warming pattern.
</item 84>

<item 85>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
The inflation bill is responsible for about a quarter of that projected decrease.)

“He brags about how he thinks we should consider him a climate champion because he reentered the Paris accord. That Paris Agreement will take us to about 3
degrees Celsius of global heating,” Kalmus said. “I don’t think we’ll have a civilization at 3 degrees Celsius.”

(A 2022 United Nations report estimated global temperatures would rise between 2.1 and 2.9 degrees Celsius by 2100 if countries held to their climate
commitments. Many countries remain off that pace.)

Climate change is weighing on scientists, but also everyday Americans.

A 2022 poll found almost two-thirds of Americans say they have been affected by extreme weather they believe was at least partially due to climate change.

About 27% of Americans say they are “very worried” about climate change; another 27% just avoid the subject as best they can. One in 10 reported feeling
symptoms of anxiety or depression over climate change.
</item 85>

<item 86>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The paleorecord could further be crucial for understanding the timescales and relative importance of internal variability, forced response, and extratropical
forcing to rates of warming or cooling in the equatorial Pacific in nature and in climate models.

Should We Trust Model Patterns in Climate Change Scenarios?

Our understanding of the pattern effect raises the question of whether climate models can reproduce observed warming patterns. Coupled climate models simulate a
freely evolving ocean surface and hence have their own expression of internal variability. For example, we would not expect a coupled model to simulate El Niño
events at the same time as they happened in nature, except by chance. This feature complicates comparisons between models and observations, especially on
decadal timescales and beyond, for which we know little about the spectrum of internal variability in the real world.

Unfortunately, coupled climate models seem unable to simulate observed surface warming patterns across some key regions in which surface sensitivity modulates
clouds, even when they account for internal variability.
</item 86>

<item 87>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
In other words, first, we need to improve our understanding of drivers of sea surface temperature patterns—decadal coupled variability; the pace and spatial
structure of ocean heat uptake; and forcing by aerosols, greenhouse gases, and volcanoes—as well as the relative timing of these drivers. An important goal is
to explain historical patterns quantitatively and to predict their likely future evolution. Second, we need to quantify the dependence of local and remote
top-of-the-atmosphere radiative fluxes on the magnitude, spatial scale, and sign (positive or negative) of surface temperature changes. Third, we have to
explain how ocean heat uptake and radiative feedback are connected on various timescales, both globally and regionally.

Interpreting Paleorecords Requires Caution

In the same way that the historical evolution of surface warming and the planet’s radiation budget inform Earth’s future only to a limited extent, analyses of
paleorecords for recent and deep time intervals must account for the pattern effect to be applicable to climate projections of the future.
</item 87>

<item 88>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
However, the challenge lies in deciphering which elements of the Mediterranean lifestyle contribute most significantly to improved health. Sotos-Prieto and her
team designed and developed a Mediterranean lifestyle index called MEDLIFE using a questionnaire that calculated an individual’s adherence to a Mediterranean
way of life. This index considers three primary components: food consumption, dietary habits, and physical activity, including rest, social habits, and
conviviality.

This type of analysis allowed the researchers to dissect the impact of each of these components, even drilling down to specific items like “hours of sleep” or
“socializing with friends”. Moreover, the MEDLIFE index could help capture synergistic effects of multiple components that may not be apparent when examining
individual components in isolation.

In their study, the team evaluated the impact of a Mediterranean-like lifestyle over the course of ten years in more than 110,000 participants located in the
UK.
</item 88>

<item 89>
'We know far more about the deep ocean than the moon or Mars,' says explorer Jon Copley (Live Science: The Most Interesting Articles, 2023-09-28T11:01:16+00:00)
How worried are you about that?

JC: I think it's great that deep sea mining has made people care more about the deep ocean, but it hasn't actually started yet and research does not support
some of the more hyperbolic headlines.

Related: More than 5,000 new species found in 'pristine' deep-sea wilderness. But they could soon be wiped out.

There's a lot of research focused on how we're going to manage mining, if it does go ahead. And there are some habitat types in the deep ocean that we don't
need to do further research on, because we know they are so vulnerable. We know that we would risk species extinction at active hydrothermal vents, for example,
because they're a tiny habitat globally — just 50 square kilometers [19 square miles] — with more than 400 animal species not found in any other habitat type.
</item 89>

<item 90>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
For carbon capture, Persson imagines fitting MXene filters atop chimneys where the exhaust from burned materials is rich in CO 2 .

To absorb CO 2 from the air elsewhere, Thakur suggests putting stacks of MXene screens out in the open. This method would be less efficient at capturing CO 2 ,
though, since average levels of CO 2 in the air far from pollution sources are quite low.

Keep in mind, trapping CO 2 anywhere is just a first step in managing this pollutant, says Vitalie Stavila. He’s a materials chemist at Sandia National
Laboratories in Livermore, Calif. “The problem is, what are you going to do with all that CO 2 ?” he says. Scientists will have to figure out how to retrieve
the trapped gas from the MXenes after they’re full.

MXenes could be heated up to release their CO 2 for long-term storage somewhere else.
</item 90>

<item 91>
Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. (Engadget, 2023-10-01T14:30:34+00:00)
To understand the dynamic, we need to take a closer look at the concept of energy democracy, as generally understood by the climate movement, and its tenets:
renewable, small-scale systems, and community control. The bitter irony of the current moment is that it’s not just rising authoritarianism that is blocking us
from good futures. It’s also our narrow and warped conceptions of democracy that are trapping us.

This article contains affilate links; if you click such a link and make a purchase, we may earn a commission.
</item 91>

<item 92>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
BELFAST, Maine — How does it feel to be a climate scientist in 2023?

Let’s set the scene. Hundreds of picnickers are sprawled across a lawn speckled with colorful blankets. Yachts bob in the bay. Guitars blare, children wrestle
and dancers twirl across a concert lawn in the golden glow of a dying summer night.

It’s a picture-perfect August evening for the seaside town of Belfast’s “Summer Nights” concert series.

Off to the side, Peter Kalmus sits alone. His eyes are closed. His toes are tucked beneath his body. He’s catching an occasional side-eye from onlookers.

“When I’m meditating, I don’t feel anxious,” Kalmus later says, explaining that he strives for two hours of meditation a day to keep his climate anxiety at bay.
Otherwise, “it’s completely overwhelming.”

Many of us avoid thinking too deeply about the worst scenarios of climate change. He is fixated on them.
</item 92>

<item 93>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
This is another in our series of stories identifying new technologies and actions that can slow climate change, reduce its impacts or help communities cope with
a rapidly changing world.

One up-and-coming class of tiny materials could help solve the huge problem of climate change. How? By catching climate-warming carbon dioxide out of the air.

The nano-scale materials are called MXenes (MAX-eenz). Their name comes from their structure, which is made up of alternating layers of M and X atoms. “M”
layers consist of some transition metal. (Those atoms are bulky and can shed a variable number of electrons.) “X” layers contain atoms of another element, such
as carbon or nitrogen.

A single MXene sheet contains a few of these alternating M and X layers chemically bound to each other. It may look like MXM or MXMXM. Each sheet is typically
only about 1 nanometer thick. For comparison, a sheet of paper is 100,000 times thicker.
</item 93>

<item 94>
A scientist reckons with climate grief (Yahoo News, 2023-12-24T13:06:02+00:00)
He has been on a 15-year quest to halt climate change and has already oriented most everything in his life to minimize his own climate footprint and maximize
what he can do to push society to change.

Nothing has been enough.

And so this summer, as extreme heat and rainfall battered many American communities, Kalmus visited Maine searching for something different. He was looking not
for climate solutions, but for new ways to withstand climate change’s harms — and also to make peace with the grief that often feels inescapable for those who
reckon with global warming daily.

climate scientist exercise bicycle environmentalist (Evan Bush / NBC News)

That search brought him to an unusual community called the Possibility Alliance, a homestead where a small group of people is trying to figure out how to live
in a world headed for a breakdown (as they see it).

Its leaders eschew capitalism, grow much of their own food, and live without electricity. They’ve dedicated their lives to protest, living off the land as
simply as possible, and are preparing their community for ecological disaster and to host climate refugees.
</item 94>

<item 95>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
This diagram depicts a process to convert MAX phase materials into thin sheets of MXenes. An acid etches out all but the M layers (blue) and X layers (purple)
to yield thick MXene sheets. The far right panel shows a scanning electron microscope image of the MXene. M. Ozkan et al/Chem. Oct. 4, 2023 ( CC BY 4.0 DEED

MXenes have “probably the largest diversity and tunability any 2-D material can offer,” says Michael Naguib. By tunable, this engineering physicist means that
MXenes can be tweaked to take on different properties. Naguib works at Tulane University in New Orleans, La. He made the world’s first MXene in 2011, a titanium
carbide.

Scientists have now made more than 50 MXenes. New ones arise when scientists mix and match different M and X atoms. Each MXene has slightly different
properties, from how well it conducts electricity to how strong it is.
</item 95>

<item 96>
Patterns of Surface Warming Matter for Climate Sensitivity (Eos: Earth And Space Science News, 2023-10-31T12:03:09+00:00)
The most pressing question is whether climate model simulations will be as far off from observations in the future as they have been relative to recent past
conditions (compare Figure 2, left and right). If this turns out to be the case, how will model biases in surface warming trends be reflected in radiative
feedbacks and global mean warming rates? We need to quantify whether coupled models compensate for their bias in the surface warming pattern and radiative
feedbacks through erroneous ocean heat uptake rates or aerosol forcings.

The advent of global coupled storm- and cloud-resolving models, which currently can simulate only a few months to years at a time, presents an exciting
opportunity. How do we evaluate surface warming patterns and radiative feedbacks in these simulations and compare them meaningfully to observations and coarser
resolution models, knowing that even several decades are not enough to robustly detect a forced trend in some regions?
</item 96>

<item 97>
Gen Z just saw the worst war in Israel in 50 years break out all over the internet, but they were raised on 'disturbing images' (Fortune,
2023-10-10T20:17:09+00:00)
But they’re not just reading about the detrimental effects of human-caused global warming, they’re living through the consequences themselves.

This summer reached record-breaking temperatures, with July being the hottest month the planet has seen in over 100,000 years. As a result, Arizona experienced
a monthlong heat wave with temperatures at or above 110 degrees every day. Deadly fires broke out across the Mediterranean; suffocating smoke from Canadian
wildfires blanketed New York City and the Northeast for days; and ice melt in the Arctic accelerated.

And extreme heat is likely here to stay—and get worse—unless countries can rapidly reduce their carbon emissions. That’s why Gen Z is more concerned with
sustainability than any generation before them.
</item 97>

<item 98>
New ultrathin materials can pull climate-warming CO2 from the air (Science News For Students, 2023-12-15T11:30:00+00:00)
A common one is hydrofluoric (Hy-droh-FLOOR-ik) acid. (This acid is so harsh that it can dissolve the minerals in your bones!) The treatment leaves behind just
the M and X layers.

Since the materials’ discovery, scientists have found safer ways to make MXenes. These techniques use milder acids or bases. One such chemical is sodium
hydroxide. That’s an ingredient in soap. Another recent method is to use molten fluoride salts to etch away the A layer in MAX phases. No harsh compounds are
needed there, although it does take high temps — up to 750° Celsius (1,380° Fahrenheit).

MXenes are the largest group of flat, sheetlike nanomaterials. Scientists refer to them as being largely two-dimensional, or 2-D. Another famous example of a
2-D material is graphene.

This diagram depicts a process to convert MAX phase materials into thin sheets of MXenes.
</item 98>

<item 99>
Bringing aqueous rechargeable zinc iodine batteries to the mainstream energy market (Advanced Science News, 2023-09-26T11:50:40+00:00)
One of the major drawbacks of these lithium-ion batteries is the inherent risk of burning or even explosion as the result of thermal runaway, with overheated
lithium-ion batteries deemed responsible for a number of fires in the past.

A possibility with zinc

One potential substitute for lithium-ion batteries are rechargeable zinc iodine batteries , which have the advantage of being safer, costing less, and being
more friendly to the environment than lithium alternatives.

However, zinc iodine batteries are not perfect. Composed of an negative iodine electrode and a positive zinc electrode , these batteries suffer from issues like
lower energy output compared to commercial non-aqueous lithium batteries and the growth of zinc dendrites on the anodes — projections of material that build on
electrodes and short batteries when they cross from one electrode to another — as well as the corrosion of the zinc anode that result in reduced efficiency and
even short-circuiting.
</item 99>

<item 100>
What about the Mediterranean lifestyle reduces mortality? Scientists find out (Advanced Science News, 2023-09-26T09:49:27+00:00)
“One way to start could be to emphasize the use of fresh, local products and ingredients in Mediterranean-style meals,” said Sotos-Prieto. “[But] changing
dietary habits and lifestyle is a gradual process, and affordability is also an important question.”

Other questions remain as to how governments should invest in introducing new healthy habits, which ones should be given preference, and how much change is it
enough to start seeing benefits.

The task is not simple since culture can define anything from ingredients and recipes to different social habits and work-life schedules. But Sotos-Prieto is
optimistic.

“I think that picking realistic goals, even if small, and sustaining them over time can make a difference [in public health],” concluded Sotos-Prieto.

Reference: Maroto-Rodriguez J. et al., Association of a Mediterranean Lifestyle With All-Cause and Cause-Specific Mortality: A Prospective Study from the UK
Biobank, Mayo Clinic proceedings (2023).
</item 100>
=== END ITEM id=04090cd24b8c13c8 ===

=== ITEM id=aae2fdbe67ec8f98 ===
Gold sentences and facts:
Sentence 0: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 0.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 0.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 0.2: The case argues that Google siphons off news publishers' readers.
  Fact 0.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 0.4: The case argues that Google does this through anticompetitive means.
Sentence 1: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 1.0: Wired removed an article for not meeting its standards.
  Fact 1.1: The article was shared widely on social media.
  Fact 1.2: The article was boosted in write-ups at other outlets.
  Fact 1.3: The degree to which the article was shared and boosted shows how willing people are to accept foul play on Google's part.

Retrieved items:
<item 1>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"I was disappointed that she doesn't talk before or after her shows because she has to save her voice for the 44 songs she sings ... She doesn't meet anybody,
or at least she didn't want to meet me, so I took it personal," Kelce said on an episode of the "New Heights" podcast he does with his brother, Philadelphia
Eagles center Jason Kelce.

He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in
Swift. When his brother asked if it was his jersey number or his phone number, Kelce replied, "you know which one."

"If you're up on Taylor Swift concerts, there are friendship bracelets and I had received a bunch of them, but I wanted to give Taylor Swift one with my number
on it."

Aug. 5: Kelce names his celebrity crush

Well before the two were rumored to be dating, Kelce was asked his crush.
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Here's a look at how it all started and where it has gone since:

June 29: Kelce makes friendship bracelets for fellow TEs

Is this where it all begin? Kelce made a friendship bracelet to give to the newcomers at a camp for NFL tight ends. He said he was "taking a page out of Taylor
Swift's book."

July 26: Kelce makes friendship bracelet for Swift

"So make the friendship bracelets, take the moment and taste it." - You're On Your Own Kid

Kelce announced that he went to Swift's highly anticipated concert, but said he did not have the chance to meet the 12-time Grammy winner.
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
We will see what happens in the near future."

Sept. 22: Another bracelet

"Lately [he's] been dressing for revenge." - Vigilante Sh--

Kelce was spotted at his nonprofit charity event wearing a bracelet that says "Vigilante Sh--," the name of one of Swift's songs.

Sept. 24: Swift attends Chiefs game

"Dating the boy on the football team." - Fifteen

Swift took Kelce up on his offer and attended the Chiefs' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering
and even chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0.
Kansas City went on to win 41-10 to improve to 2-1.

Sept. 24: Swift mingles with Donna Kelce



"Talkin' over a football team."
</item 3>

<item 4>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."
</item 4>

<item 5>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Head coach Andy Reid was also asked about his star tight end's supposed relationship, to which he quipped, "I set them up."

Sept. 24: Kelce and Swift leave stadium together

"You know when it's time to go." - It's Time To Go

This was the first time the two were photographed together. They walked side by side, Kelce fittingly wearing an outfit with "1989" in the title and Swift
wearing a Chiefs zip-up.

Sept. 24: The two leave in a convertible

"Nothing good starts in the getaway car." - Getaway Car

Swift and Swift were spotted in a convertible, with the tight end driving, after the Chiefs' win. Kelce reportedly rented out a restaurant for a private party
with Swift and his teammates.

"Taylor Swift likes that Travis Kelce 'pursued her' and the two are having a 'great time getting to know each other,'" a source told Entertainment Tonight.
</item 5>

<item 6>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"'Taylor wants a guy that's into his career, does his own thing, but is also supportive and understands her,' the source added, "and Travis fits those
qualities.'"

Sept. 25: Kelce jersey sales skyrocket

Call it what you want; I'll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey sales, his podcast
is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game last week, according
to Front Office Sports.

There has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.

Sept. 25: Belichick weighs in on the matter



New England Patriots head coach does not always comment on situations that take place in the game, but he was fully ready with a sound bite on the Kelce-Swift
situation.
</item 6>

<item 7>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 7>

<item 8>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.
</item 8>

<item 9>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But before he could even answer, the answer was given for him.

Sept. 14: Rumors aren't denied

"The rumors are terrible and cruel, but honey, most of them are true." - New Romantics

Of course people were going to ask Jason, who is very close with his brother, whether rumors of the two hanging out were true. He did not confirm them, but he
also did not deny them.

Sept. 17: The references begin

"I watched it begin again." - Begin Again

The rumors began to pick up, and now there were a lot of people, commentators included, making Swift references around Kelce.

When the tight end caught a touchdown pass from Patrick Mahomes, CBS Sports' Ian Eagle said, "Kelce finds a blank space for the score."

This was only the start to many, many, Swift lyrics being referenced during NFL games.

Sept. 21: Kelce sends Swift an invite

"So inviting, I almost jump in."
</item 9>

<item 10>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
21: Kelce sends Swift an invite

"So inviting, I almost jump in." - Gold Rush

While on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.

"It is hilarious how much traction this has actually got," he said of all the talk. "It's like that old game in school called telephone where everybody is just
whispering in each others ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."

He then admitted that he invited Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the
Chiefs.

"I threw it out there, I threw the ball in her court and I told her you know, I've seen you rock the stage in Arrowhead you might have to come see me rock the
stage in Arrowhead and see which one is a little more lit. We will see what happens in the near future."
</item 10>
=== END ITEM id=aae2fdbe67ec8f98 ===
